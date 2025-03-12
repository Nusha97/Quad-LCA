"""
SYNOPSIS
    Contains functions required to load data, train and run inference on
    neural network models for learning value function from trajectories.
DESCRIPTION

    Implementation uses JAX libraries (flax) to define functions to train,
    evaluate and save deep learning models.
AUTHOR

    Anusha Srikanthan <sanusha@seas.upenn.edu>
LICENSE

VERSION
    0.0
"""
from flax import linen as nn
from flax.training import checkpoints  # need to install tensorflow
import torch.utils.data as data
import numpy as np
import optax

## Progress bar
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from torch.utils.data import Dataset
import torch
import cvxpy as cp
from jax.scipy.optimize import minimize
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_affine_set
from jax import grad, jit
from jaxopt import LBFGS
from torch.utils.tensorboard import SummaryWriter
import scipy.linalg as spl
from trajgen.trajutils import _cost_matrix, _diff_coeff


writer = SummaryWriter()


class TrajDataset(Dataset):
    """
    Dataset class inherited from torch modules
    """
    def __init__(self, xtraj, utraj, rtraj, xtraj_):
        """
        Input:
            - xtraj:        np.array(N, p), sequence of states
            - utraj:        np.array(N, q), sequence of inputs
            - rtraj:        np.array(N,), sequence of rewards
            - xtraj_:       np.array(N, p), sequence of next states
        """
        self.xtraj = xtraj
        self.utraj = utraj
        self.rtraj = rtraj
        self.xtraj_ = xtraj_

    def __len__(self):
        return len(self.xtraj)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.xtraj[idx], self.utraj[idx], self.rtraj[idx], self.xtraj_[idx]


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def calculate_loss(state, params, batch):
    data_state, data_input, data_cost, data_next = batch
    pred = state.apply_fn(params, data_state)
    target = data_cost

    # Calculate the loss
    loss = optax.l2_loss(pred.ravel(), target.ravel()).mean()

    return loss


@jax.jit  # Jit the function for efficiency
def train_step(state, batch):
    # Gradient function
    grad_fn = jax.value_and_grad(calculate_loss,  # Function to calculate the loss
                                 argnums=1,  # Parameters are second argument of the function
                                 has_aux=False  # Function has additional outputs, here accuracy
                                )
    # Determine gradients for current model, parameters and batch
    loss, grads = grad_fn(state, state.params, batch)
    print(loss)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss


@jax.jit  # Jit the function for efficiency
def eval_step(state, batch):
    # Determine the accuracy
    loss = calculate_loss(state, state.params, batch)
    return loss


def train_model(state, data_loader, num_epochs=100):
    # Training loop
    count = 0
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        for batch in data_loader:
            count += 1
            state, loss = train_step(state, batch)
            # We could use the loss and accuracy for logging here, e.g. in TensorBoard
            # For simplicity, we skip this part here
            epoch_loss += loss
            writer.add_scalar('Train loss', np.array(epoch_loss), count)
    return state


def eval_model(state, data_loader, batch_size):
    all_losses, batch_sizes = [], []
    for batch in data_loader:
        batch_loss = eval_step(state, batch)
        all_losses.append(batch_loss)
        batch_sizes.append(batch[0].shape[0])
    # Weighted average since some batches might be smaller
    loss = sum([a*b for a, b in zip(all_losses, batch_sizes)]) / sum(batch_sizes)
    # writer.add_scalar('Train batch loss', np.array(loss), count)
    print(f"Loss of the model: {loss:4.2f}")


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir, step=0):
    checkpoints.save_checkpoint(workdir, state, step)


def coeff2traj(coeffs, order, ts, numsteps):
    ''' Constructs a trajectory from polynomial coefficients
    Input:
        - coeffs:       np.array(p, #segments, polynomial order)
        - ts:           np.array(#segments+1), timestamps
        - numsteps:     Integer, number of time steps in the resulting traj
    Return:
        - traj:         np.array(numsteps), trajectory from coeffs
    '''
    # p = coeffs.shape[0]
    p = 4
    ref = jnp.zeros([p, numsteps])
    times = np.linspace(ts[0], ts[-1], numsteps)
    k = 0
    for i, tt in enumerate(times):
        if tt > ts[k+1]: k += 1
        # ref = ref.at[:, i].set(_diff_coeff(coeffs.shape[2]-1, tt-ts[k], 0) @ coeffs[:,k,:].T)
        # print(coeffs[:, k::order+1].shape)
        # print(_diff_coeff(order, tt-ts[k], 0).shape)
        ref = ref.at[:, i].set(_diff_coeff(order, tt-ts[k], 0) @ coeffs[:, k::order+1].T)
    return times, ref


def linear_map_coeff(ts, order, num_steps, p, x0, regularizer, coeff, value):
    durations = ts[1:] - ts[:-1]
    cost_mat = spl.block_diag(*[_cost_matrix(order, 3, d) for d in durations])
    cost = 0
    print("Cost mat", cost_mat.shape)
    for pp in range(p):
        # cost += jnp.dot(coeff[pp].reshape(-1), cost_mat @ coeff[pp].reshape(-1))
        # cost = 0
        cost += jnp.dot(coeff[pp, :], cost_mat @ coeff[pp, :])

    ref = coeff2traj(coeff, order, ts, num_steps)[1].ravel()
    print("ref shape", ref.shape)
    # ref = jnp.zeros([p, num_steps])
    # cost += regularizer.pred(x0, ref)[0]
    print(jnp.exp(regularizer.apply_fn(regularizer.params, jnp.append(x0, ref))))
    cost += value * jnp.exp(regularizer.apply_fn(regularizer.params, jnp.append(x0, ref))).ravel()[0]

    return cost


def gradient_descent(func, proj, init, goal, state, params, init_params, learning_rate=0.001, num_iters=100):
    # Define the gradient of the function using JAX's `grad` function
    grad_func = grad(func)

    # Define a JIT-compiled version of the gradient descent update step
    @jit
    def update(ref, i):
        gradient = grad_func(ref, init, goal, state, params)
        # project the gradient
        gradient = proj.T @ gradient
        return ref - learning_rate * gradient

    # Perform gradient descent by iteratively updating the parameters
    ref = init_params
    for i in range(num_iters):
        ref = update(ref, i)

    return ref


def calculate_cost(ref, init, goal, state, params):
    pred = state.apply_fn(params, jnp.append(init, ref)).ravel()
    len = ref.shape[0]
    sum = 0
    for i in range(3, len-3):
        sum += jnp.linalg.norm(ref[i:i + 3] - ref[i - 3:i]) ** 2
    return pred[0] #sum #+ jnp.linalg.norm(data_state[0:3] - init) ** 2 # + jnp.linalg.norm(data_state[-3:] - goal) ** 2  # Adding terminal state constraint to the objective function


def test_model(trained_state, data_loader, batch_size):
    """
    At test time, we optimize for a reference trajectory that minimizes LQR cost + regularizer
    Need to add penalty for regularizer and change cost
    :return:
    """
    data_state, _, data_cost, _ = next(iter(data_loader))
    solution = []
    orig = []
    jax_sol = []

    def calc_cost_GD(ref):
        pred = trained_state.apply_fn(trained_state.params, ref).ravel()
        return pred[0]

    for data, cost in zip(data_state, data_cost):
        # print("Cost computed using the network", calculate_cost(data, trained_state, trained_state.params))
        # print("True cost", cost)
        print("Shape of data", data.shape)
        # pg = ProjectedGradient(calculate_cost, data[3:], args=(data[0:3], data[-3:], trained_state, trained_state.params), projection=projection_non_negative)
        # pg.run()

        """w_init = data[3:]
        lbfgsb = ScipyBoundedMinimize(fun=calculate_cost, data[3:], args=(data[0:3], data[-3:], trained_state, trained_state.params), method="l-bfgs-b")
        lower_bounds = jnp.zeros_like(w_init)
        upper_bounds = jnp.ones_like(w_init) * jnp.inf
        bounds = (lower_bounds, upper_bounds)
        lbfgsb_sol = lbfgsb.run(w_init, bounds=bounds, data=(data[:3], cost)).params
        jax_sol.append(lbfgsb_sol)"""
        A = np.zeros((6, 18))
        A[0, 0] = 1
        A[1, 1] = 1
        A[2, 2] = 1
        A[-3, -3] = 1
        A[-2, -2] = 1
        A[-1, -1] = 1
        b = np.append(data[:3], data[-3:])
        PGD = ProjectedGradient(calc_cost_GD, projection_affine_set)
        solution.append(PGD.run(data, hyperparams_proj=(A, b)))
        #GD = LBFGS(calculate_cost)
        #solution.append(GD.run(data[3:], data[0:3], data[-3:], trained_state, trained_state.params))
        # solution.append(minimize(calculate_cost, data[3:], args=(data[0:3], data[-3:], trained_state, trained_state.params), method="BFGS"))
        # solution.append(ScipyMinimize(calculate_cost, method="BFGS", data[3:], data[0:3], data[-3:], trained_state, trained_state.params))
        orig.append(data)
    # Return new reference and cost
    return jax_sol, solution, orig




