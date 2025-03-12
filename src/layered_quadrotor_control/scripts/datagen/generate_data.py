"""
SYNOPSIS
    A simple trajectory generator code for nonlinear dynamical systems
DESCRIPTION

    Generates multiple trajectories for the given nonlinear system using ILQR
    based on specified parameters. Currently, no constraints on inputs have been
    implemented.
AUTHOR

    Anusha Srikanthan <sanusha@seas.upenn.edu>
LICENSE

VERSION
    0.0
"""

import functools

import jax.numpy as np
import numpy as onp
import jax

from trajax import optimizers
from trajax.integrators import euler
from trajax.integrators import rk4

import matplotlib.pyplot as plt
import pickle
from helper_functions import angle_wrap, compute_rdot

from itertools import accumulate

gamma = 1


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(str):
    with open(str, 'rb') as handle:
        return pickle.load(handle)


class ILQR():
    """
    Apply iterative LQR from trajax on any specified dynamical system
    """

    def __init__(self, dynamics, maxiter=1000):
        self.maxiter = maxiter
        self.dynamics = dynamics
        self.constraints_threshold = 1.0e-1
        self.goal = None
        self.horizon = None


    def discretize(self, type):
        if type == 'euler':
            return euler(self.dynamics, dt=0.01)
        else:
            return rk4(self.dynamics, dt=0.01)


    def apply_ilqr(self, x0, U0, goal, maxiter=None, true_params=(100, 1.0, 0.1)):
        """
        Function to execute ilqr from trajax to generate reference trajectories
        :param true_params: Specify cost weights and gain constants of PD controller
        :return: X, U, total_iter
        """
        dynamics = self.discretize('rk4')
        m = x0.shape[0]

        if maxiter:
            self.maxiter = maxiter

        self.goal = goal

        self.horizon = U0.shape[0]

        if m > 2:
            key = jax.random.PRNGKey(75493)
            true_param = (2000 * np.eye(m), np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1]]), 0.1 * np.eye(3))
            final_weight, stage_weight, cost_weight = true_param
        else:
            final_weight, stage_weight, cost_weight = true_params

        def cost(params, state, action, t):
            """
            Cost function should be designed assuming the state (x) and action (r, rdot)
            :param params: weight matrices and PD gain constants
            :param state: x, r
            :param action: rdot
            :param t: current time step
            :return: List of state, input, total_iter
            """
            final_weight, stage_weight, cost_weight = params

            if m == 2:
                state_err = state[0] - np.squeeze(action[0])
                state_cost = stage_weight * state_err ** 2
                action_cost = np.squeeze(action[1]) ** 2 * cost_weight
                terminal_cost = final_weight * (state[0] - self.goal) ** 2
            else:
                state_err = state[0:3] - state[3:]
                state_cost = np.matmul(np.matmul(state_err, stage_weight), state_err)
                action_cost = np.matmul(np.matmul(np.squeeze(action), cost_weight), np.squeeze(action))
                terminal_err = state - self.goal
                terminal_cost = np.matmul(np.matmul(terminal_err, final_weight), terminal_err)

            return np.where(t == self.horizon, terminal_cost, state_cost + action_cost)


        def equality_constraint(x, u, t):
            del u

            # maximum constraint dimension across time steps
            dim = 3

            def goal_constraint(x):
                err = x[0:3] - self.goal[0:3]
                return err

            return np.where(t == self.horizon, goal_constraint(x), np.zeros(dim))

        X, U, _, _, _, _,_, _, _, _, _, total_iter = optimizers.constrained_ilqr(functools.partial(cost, true_param), dynamics, x0,
                                    U0, equality_constraint=equality_constraint, constraints_threshold=self.constraints_threshold, maxiter_al=self.maxiter)
        #X, U, _, _, _, _, total_iter = optimizers.ilqr(functools.partial(cost, true_params), dynamics, x0, U0, self.maxiter)
        return X, U, total_iter


def generate_polynomial_trajectory(start, end, T, order):
    """
    Generates a polynomial trajectory from start to end over time T
    start: start state
    end: end state
    T: total time
    order: order of the polynomial
    """
    # Define the time vector
    t = onp.linspace(0, 1, T)
    n = start.shape[0]

    # Solve for the polynomial coefficients
    coeffs = onp.zeros((order + 1, n))
    for i in range(n):
        coeffs[:, i] = onp.polyfit(t, t * (end[i] - start[i]) + start[i], order)

    # Evaluate the polynomial at the desired time steps
    polynomial = onp.zeros((T, n))
    for i in range(n):
        polynomial[:, i] = onp.polyval(coeffs[::-1, i], t)
    trajectory = polynomial + start

    return trajectory


def gen_uni_training_data(lqr_obj, num_iter, state_dim, inp_dim, goals=None, inits=None):
    """
    Generate trajectories for training data
    :param dynamics: Dynamics function to be passed to ILQR
    :param goals: Set of goals generated using random generator
    :param inits: Set of initial conditions to be tested on
    :return: xtraj, rtraj, rdottraj, costs
    """

    horizon = 10
    dt = 0.01

    xtraj = []
    rtraj = []
    rdottraj = []
    costs = []

    if goals == None:
        key = jax.random.PRNGKey(89731203)
        goal_xy = jax.random.uniform(key, shape=(num_iter, 2), minval=1, maxval=3)
        goal_theta = onp.zeros(shape=(num_iter,))
        goal = np.append(goal_xy, goal_theta)
        print(goal)
        goals = np.reshape(goal, (num_iter, int(state_dim/2)), order='F')
        print(goals)

    if inits == None:
        key = jax.random.PRNGKey(95123459)
        init_xy = jax.random.uniform(key, shape=(num_iter, 2), minval=0, maxval=2)
        # init_theta = onp.zeros(shape=(num_iter,))
        init_theta = jax.random.uniform(key, shape=(num_iter, ), minval=0, maxval=np.pi)
        init = np.append(init_xy, init_theta)
        inits = np.reshape(init, (num_iter, int(state_dim/2)), order='F')
        print(inits)

    for j in range(num_iter):

        tk_cost = 100
        it = 2
        while tk_cost/horizon >= 0.1:
            if it > 1000:
                break
            x0 = np.append(inits[j, :], inits[j, :])
            # U0 = np.zeros(shape=(horizon, inp_dim))
            U0 = np.zeros(shape=(horizon-1, inp_dim))
            # U0 = np.array([np.polyval(np.array([(goals[j, 0]-inits[j, 0])/horizon, inits[j, 0]]), np.arange(horizon)),
            #                np.polyval(np.array([(goals[j, 1]-inits[j, 1])/horizon, inits[j, 1]]), np.arange(horizon)),
            #               jax.random.uniform(key, shape=(horizon,), minval=-np.pi, maxval=0)])
            if state_dim == 1:
                goal = goals[j, 0]
                U = np.append(np.array([0, inits[j, 1]]), U0)
            else:
                goal = np.append(goals[j, :], goals[j, :])
                U = np.append(U0, goals[j, :])
                U = np.reshape(U, (horizon, inp_dim))
                # U = U0.T
            x, u, t_iter = lqr_obj.apply_ilqr(x0, U, goal, it)
            it *= 5

            r_int = x[:-1, 3:] + u * dt
            r_int = onp.vstack([x0[0:3], r_int])
            tk_cost = np.linalg.norm(x[:, 0:3] - r_int)



            if tk_cost > 500:
                continue

            xtraj.append(x[:, 0:3])
            rtraj.append(r_int)
            rdottraj.append(onp.vstack([inits[j, :], u]))
            print(len(rdottraj))
            costs.append(tk_cost)

            print(tk_cost)

        # Generate polynomial trajectory with an order
        """r_poly = generate_polynomial_trajectory(inits[j, :], goals[j, :], horizon + 1, 3)
        # r_poly = np.append(r_poly)
        # r_poly = np.reshape(r_poly, (horizon+1, 3), order='F')
        # print(r_poly)
        rtraj.append(r_poly)
        c, true_x = forward_simulate(inits[j, :], r_poly, horizon)
        xtraj.append(true_x)
        rdottraj.append(compute_rdot(r_poly, dt))
        tk_cost = np.linalg.norm(true_x - r_poly)

        print(tk_cost)"""

    return xtraj, rtraj, rdottraj, costs


def unicycle_K(x, u, t):
    """
    Unicycle system controlled only by a P controller
    :param x: States of the dynamical system (x, y, theta)
    :param u: Control input (v, w)
    :param t: time horizon
    :return: xdot
    """
    del t
    px, py, theta = x[0:3]
    theta = angle_wrap(theta)
    state_dim = 6
    input_dim = 3

    x_dev = x[0:3] - x[3:]

    F = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    Kp = 5 * np.array([[2, 1, 0], [0, 1, 3]])
    Kd = np.linalg.pinv(F)

    return np.append(np.matmul(F, np.matmul(Kd, np.squeeze(u)) + np.matmul(Kp, x_dev)), np.squeeze(u))


def unicycle(x, u, t):
    """
    Unicycle system
    :param x: States of the dynamical system
    :param u: Control input to the system
    :param t: Total time horizon
    :return: xdot: Closed loop dynamical system of the unicycle with references
    """
    del t
    px, py, theta = x[0:3]
    theta = angle_wrap(theta)

    state_dim = 6
    input_dim = 3

    F = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    Kp = 50 * np.array([[2, 1, 0], [0, 1, 3]])
    key = jax.random.PRNGKey(793)
    Kd = 50 * jax.random.uniform(key=key, shape=(2, 3))

    x_dev = x[0:3] - x[3:]

    v1 = np.matmul(np.matmul(F, np.linalg.inv(np.eye(2) - np.matmul(Kd, F))), np.matmul(Kp, x_dev) - np.matmul(Kd, np.squeeze(u)))
    return np.append(v1, np.squeeze(u))



def forward_simulate(x0, r, N):
    """
    Simulate the unicycle dynamical system for the given reference trajectory
    :param x0: initial condition
    :param r: reference trajectory
    :param N: horizon of reference
    :return:
    """
    # Compute rdot numerically
    dt = 0.01
    cur_ref = r[1:]
    prev_ref = r[:-1]

    rdot = onp.zeros(r.shape)
    x = onp.zeros(r.shape)
    xdot = onp.zeros(r.shape)
    x[0, :] = x0

    rdot[1:, :] = (cur_ref - prev_ref)/dt
    # v, w = compute_input(x0, r[0, :], rdot[0, :], Kp, Kd)
    # xdot[0, :] = np.array([v * np.cos(x0[2]), v * np.sin(x0[2]), w])
    temp = unicycle_K(onp.append(x0, r[0, :]), r[0, :], 10)
    xdot[0, :] = temp[0:3]
    for i in range(1, N):
        x[i, :] = x[i-1, :] + xdot[i-1, :] * dt
        # v, w = compute_input(x[i, :], r[i, :], rdot[i, :], Kp, Kd)
        # xdot[i, :] = np.array([v * np.cos(x[i, 2]), v * np.sin(x[i, 2]), w])
        temp = unicycle_K(onp.append(x[i, :], r[i, :]), r[i, :], 10)
        xdot[i, :] = temp[0:3]

    cost = onp.linalg.norm(x[:, :2] - r[:, :2], axis=1) ** 2 + angle_wrap(x[:, 2] - r[:, 2]) ** 2
    # Computing the cumulative cost with gamma
    return list(accumulate(cost[::-1], lambda x, y: x * gamma + y))[-1], x

