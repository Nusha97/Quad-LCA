import numpy as np
from jaxopt import ProjectedGradient, GradientDescent, NonlinearCG, BacktrackingLineSearch
from jaxopt.projection import projection_affine_set
import jax.numpy as jnp
from jax import grad, jit
import jax
import time

jax.config.update("jax_enable_x64", True)


# def project_to_waypoints(coeffs, A, b, invA):
#     """
#     Return the projected coefficients
#     """
#     return coeffs - invA @ (A @ coeffs - b)


# def modify_reference(regularizer, cost_mat_full, A_coeff_full, b_coeff_full, coeffs):
#     """
#     Run gradient descent using line search and Armijo rule
#     """
#     def nn_cost(coeffs):
#         """
#         Function to compute trajectories given polynomial coefficients
#         :param coeffs: 4-dim polynomial coefficients (x, y, z, yaw)
#         :param ts: waypoint time allocation
#         :param numsteps: Total number of samples in the reference
#         :return: ref
#         """
#         return coeffs.T @ cost_mat_full @ coeffs + jnp.exp(regularizer[0].apply(regularizer[1], coeffs)[0])
#         # return coeffs.T @ cost_mat_full @ coeffs + jnp.exp(regularizer.apply_fn(regularizer.params, coeffs).ravel())
        
#     invA = A_coeff_full.T @ jnp.linalg.pinv(A_coeff_full @ A_coeff_full.T)
#     step_size = 0.00001
#     # step_size = 1.0
#     new_coeffs = coeffs
#     # solver = NonlinearCG(fun=nn_cost, maxiter=100)
#     # solver = GradientDescent(fun=nn_cost, maxiter=1)
#     # sol = solver.run(coeffs)
#     # new_coeffs = sol.params
#     # pred = sol.state.error
#     new_pred = nn_cost(new_coeffs)
#     pred = nn_cost(coeffs)

#     # ls = BacktrackingLineSearch(fun=nn_cost, maxiter=1, condition="strong-wolfe", decrease_factor=0.8)
#     # value, grad_fn = jax.value_and_grad(nn_cost)
#     # step_size, state = ls.run(init_stepsize=step_size, params=new_coeffs,
#     #                      value=value, grad=grad_fn)
    
#     # for i in range(10):  
#     #     # grads = grad_fn(new_coeffs)
#     #     ls.update()

#     for i in range(10):

#         grad_fn = jax.grad(nn_cost)
#         grads = grad_fn(new_coeffs)
#         if new_pred <= pred:
#             new_coeffs = new_coeffs - step_size * jnp.ravel(grads)
#         new_coeffs = project_to_waypoints(new_coeffs, A_coeff_full, b_coeff_full, invA)
#         pred = nn_cost(new_coeffs)
        

#     return new_coeffs, pred, False



def modify_reference(
    regularizer,
    cost_mat_full,
    A_coeff_full,
    b_coeff_full,
    coeff0,
    # maxiter=5
):
    """
    Running projected gradient descent on the neural network cost + min snap cost with constraints
    """
    @jit
    def nn_cost(coeffs):
        """
        Function to compute trajectories given polynomial coefficients
        :param coeffs: 4-dim polynomial coefficients (x, y, z, yaw)
        :param ts: waypoint time allocation
        :param numsteps: Total number of samples in the reference
        :return: ref
        """
        return coeffs.T @ cost_mat_full @ coeffs + jnp.exp(
            regularizer[0].apply(regularizer[1], coeffs)[0]
        )

    # Initialize ProjectedGradient with maxiter set to 1
    pg = ProjectedGradient(
        nn_cost,
        projection=projection_affine_set,
        maxiter=100,
        jit = True,
        # verbose=True,
    )

    # Run the initial step of ProjectedGradient
    start = time.time()
    sol = pg.run(coeff0, hyperparams_proj=(A_coeff_full, b_coeff_full))
    end = time.time()

    print("Total time", end - start)

    # Initialize variables to track the best solution and error
    best_solution = sol.params
    best_error = sol.state.error
    nan_encountered = np.isnan(best_error)
    # changes for clipping
    # for _ in range(100 - 1):
    #     sol = pg.update(
    #         sol.params, sol.state, hyperparams_proj=(A_coeff_full, b_coeff_full)
    #     )

    #     current_error_nan = np.isnan(sol.state.error)
    #     if current_error_nan:
    #         nan_encountered = True
    #         continue

    #     if sol.state.error < best_error:
    #         best_solution = sol.params
    #         best_error = sol.state.error
    #         print(f"New lowest ProximalGradient error: {best_error}")

    #     # Apply gradient clipping
    #     grads = grad(nn_cost)(sol.params)
    #     clipped_grads = jax.clip(grads, -1.0, 1.0)
    #     sol.params = sol.params - 0.00001 * clipped_grads

    # # If NaN error is encountered at the beginning, return immediately
    # if nan_encountered:
    #     print("Final lowest ProximalGradient error: NaN")
    #     return coeff0, best_error, nan_encountered

    # # Total iterations, adjust this number as needed
    # total_iterations = 100

    # # Iteratively update and check for the best solution
    # for _ in range(total_iterations - 1):
    #     sol = pg.update(
    #         sol.params, sol.state, hyperparams_proj=(A_coeff_full, b_coeff_full)
    #     )

    #     # Check for NaN errors
    #     current_error_nan = np.isnan(sol.state.error)
    #     if current_error_nan:
    #         nan_encountered = True
    #         continue

    #     # Update best solution if the current solution has a lower error
    #     if sol.state.error < best_error:
    #         best_solution = sol.params
    #         best_error = sol.state.error
    #         print(f"New lowest ProximalGradient error: {best_error}")

    print(f"Final lowest ProximalGradient error: {best_error}")
    return best_solution, best_error, nan_encountered


def main():
    # Test code here
    def regularizer(x):
        return jnp.sum(x**2)

    A = 4 * jnp.eye(2)
    b = 2.0 * jnp.ones(2)

    H = 10.0 * jnp.eye(2)
    # coeff, pred = modify_reference(regularizer, H, A, b, jnp.array([1.0, 0]))
    coeff = gradient_descent(regularizer, H, A, b, jnp.array([1.0, 0]))
    print(coeff)
    # print(pred)


if __name__ == "__main__":
    main()
