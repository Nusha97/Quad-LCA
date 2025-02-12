#! /usr/bin/env python3
"""
To perform inference on an experiment that collected 20 trajectories and compile all data
"""
import random

from construct_traj_list import *
import numpy as np
import matplotlib.pyplot as plt
from model_learning import restore_checkpoint, linear_map_coeff, coeff2traj

import ruamel.yaml as yaml
from flax.training import train_state
import optax
import jax
from mlp_jax import MLP, MLP_torch

from trajgen import nonlinear, quadratic, quadrotor
import rospy
from kr_tracker_msgs.msg import TrajectoryTrackerAction, TrajectoryTrackerGoal, LissajousTrackerAction, LissajousTrackerGoal

from layered_ref_control.mav_layer_interface import KrMavInterface

import pickle
import sys
import jax.numpy as onp
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_affine_set
from functools import partial

from learning import valuefunc
import torch

# from scipy.linalg.block_diag import blk_diag

PI = np.pi


def compute_coeff_deriv(coeff, n, segments):
    """
    Function to compute the nth derivative of a polynomial
    :return:
    """
    coeff_new = coeff.copy()
    for i in range(segments):  # piecewise polynomial
        for j in range(n):  # Compute nth derivative of polynomial
            t = np.poly1d(coeff_new[i, :]).deriv()
            coeff_new[i, j] = 0
            coeff_new[i, j+1:] = t.coefficients
    return coeff_new


def sampler(poly, T, ts):
    """
    Function to generate samples given polynomials
    :param coeff:
    :return:
    """
    k = 0
    ref = []
    for i, tt in enumerate(np.linspace(ts[0], ts[-1], T)):
        if tt > ts[k + 1]: k += 1
        ref.append(poly[k](tt-ts[k]))
    return ref


def compute_pos_vel_acc(Tref, nn_coeffs, segments, ts):
    """
    Function to compute pos, vel, acc from nn coeffs
    :param timesteps:
    :return:
    """
    # Compute full state
    coeff_x = np.vstack(nn_coeffs[0, :, :])
    coeff_y = np.vstack(nn_coeffs[1, :, :])
    coeff_z = np.vstack(nn_coeffs[2, :, :])
    coeff_yaw = np.vstack(nn_coeffs[3, :, :])

    pos = []
    vel = []
    acc = []
    jerk = []

    x_ref = [np.poly1d(coeff_x[i, :]) for i in range(segments)]
    x_ref = np.vstack(sampler(x_ref, Tref, ts)).flatten()

    y_ref = [np.poly1d(coeff_y[i, :]) for i in range(segments)]
    y_ref = np.vstack(sampler(y_ref, Tref, ts)).flatten()

    z_ref = [np.poly1d(coeff_z[i, :]) for i in range(segments)]
    z_ref = np.vstack(sampler(z_ref, Tref, ts)).flatten()
    pos.append([x_ref, y_ref, z_ref])

    #if len(nn_coeffs[0, 0, :] < 2):
    #    return pos, vel, acc, jerk, 0, 0

    dot_x = compute_coeff_deriv(coeff_x, 1, segments)
    xdot_ref = [np.poly1d(dot_x[i, :]) for i in range(segments)]
    xdot_ref = np.vstack(sampler(xdot_ref, Tref, ts)).flatten()

    dot_y = compute_coeff_deriv(coeff_y, 1, segments)
    ydot_ref = [np.poly1d(dot_y[i, :]) for i in range(segments)]
    ydot_ref = np.vstack(sampler(ydot_ref, Tref, ts)).flatten()

    dot_z = compute_coeff_deriv(coeff_z, 1, segments)
    zdot_ref = [np.poly1d(dot_z[i, :]) for i in range(segments)]
    zdot_ref = np.vstack(sampler(zdot_ref, Tref, ts)).flatten()
    vel.append([xdot_ref, ydot_ref, zdot_ref])

    ddot_x = compute_coeff_deriv(coeff_x, 2, segments)
    xddot_ref = [np.poly1d(ddot_x[i, :]) for i in range(segments)]
    xddot_ref = np.vstack(sampler(xddot_ref, Tref, ts)).flatten()

    ddot_y = compute_coeff_deriv(coeff_y, 2, segments)
    yddot_ref = [np.poly1d(ddot_y[i, :]) for i in range(segments)]
    yddot_ref = np.vstack(sampler(yddot_ref, Tref, ts)).flatten()

    ddot_z = compute_coeff_deriv(coeff_z, 2, segments)
    zddot_ref = [np.poly1d(ddot_z[i, :]) for i in range(segments)]
    zddot_ref = np.vstack(sampler(zddot_ref, Tref, ts)).flatten()
    acc.append([xddot_ref, yddot_ref, zddot_ref])

    dddot_x = compute_coeff_deriv(coeff_x, 3, segments)
    xdddot_ref = [np.poly1d(dddot_x[i, :]) for i in range(segments)]
    xdddot_ref = np.vstack(sampler(xdddot_ref, Tref, ts)).flatten()

    dddot_y = compute_coeff_deriv(coeff_y, 3, segments)
    ydddot_ref = [np.poly1d(dddot_y[i, :]) for i in range(segments)]
    ydddot_ref = np.vstack(sampler(ydddot_ref, Tref, ts)).flatten()

    dddot_z = compute_coeff_deriv(coeff_z, 3, segments)
    zdddot_ref = [np.poly1d(dddot_z[i, :]) for i in range(segments)]
    zdddot_ref = np.vstack(sampler(zdddot_ref, Tref, ts)).flatten()
    jerk.append([xdddot_ref, ydddot_ref, zdddot_ref])

    yaw_ref = [np.poly1d(coeff_yaw[i, :]) for i in range(segments)]
    yaw_ref = np.vstack(sampler(yaw_ref, Tref, ts)).flatten()

    dot_yaw = compute_coeff_deriv(coeff_yaw, 1, segments)
    yawdot_ref = [np.poly1d(dot_yaw[i, :]) for i in range(segments)]
    yawdot_ref = np.vstack(sampler(yawdot_ref, Tref, ts)).flatten()

    #print("Pos array shape", np.array(pos).shape)

    return np.vstack(pos), np.vstack(vel), np.vstack(acc), np.vstack(jerk), yaw_ref, yawdot_ref


def load_torch_model(trained_model_state):
    # Load checkpoint
    weights = trained_model_state.params['params']

    # Store weights of the network
    hidden_wts = [
        [weights['linear_0']['kernel'], weights['linear_0']['bias']],
        [weights['linear_1']['kernel'], weights['linear_1']['bias']],
        [weights['linear_2']['kernel'], weights['linear_2']['bias']],
    ]
    linear2_wts = [weights['linear2']['kernel'], weights['linear2']['bias']]

    def convert_torch(x):
        print(x.shape)
        return torch.from_numpy(np.array(x))

    # Create network
    inp_size = 1204
    num_hidden = [500, 400, 200]
    mlp_t = MLP_torch(inp_size, num_hidden)

    for i in range(3):
        mlp_t.hidden[i].weight.data = convert_torch(hidden_wts[i][0]).T
        mlp_t.hidden[i].bias.data = convert_torch(hidden_wts[i][1])

    mlp_t.linear2.weight.data = convert_torch(linear2_wts[0]).T
    mlp_t.linear2.bias.data = convert_torch(linear2_wts[1])
    return mlp_t


def generate_lissajous_traj(s, x_num_periods, y_num_periods, z_num_periods, yaw_num_periods, period, x_amp, y_amp, z_amp, yaw_amp):
    """
    Function to generate Lissajous trajectory
    :return:
    """
    x = lambda a: x_amp * (1 - np.cos(2 * PI * x_num_periods * a / period))
    y = lambda a: y_amp * np.sin(2 * PI * y_num_periods * a / period)
    z = lambda a: z_amp * np.sin(2 * PI * z_num_periods * a / period)
    yaw = lambda a: yaw_amp * np.sin(2 * PI * yaw_num_periods * a / period)
    return [x(s), y(s), z(s), yaw(s)]


def load_infbags(str1, str2):
    # Load inference bags with Lissajous curves
    bag = load_bag(str1)
    ref_traj, actual_traj, input_traj, cost_traj = compute_traj(bag, str2)
    bag.close()
    return ref_traj, actual_traj, input_traj, cost_traj


def compute_tracking_cost(ref_traj, actual_traj, input_traj):
    xcost = np.linalg.norm(actual_traj[:, (0, 1, 2)] - ref_traj[:, (0, 1, 2)], axis=1) ** 2
    ucost = 0.1 * np.linalg.norm(input_traj, axis=1) ** 2
    # Cost from yaw (needs special treatment because quotient norm)
    ar = np.abs(actual_traj[:, 12] - ref_traj[:, 12])
    ra = np.abs(actual_traj[:, 12] + 2 * np.pi - ref_traj[:, 12])
    yawcost = np.minimum(ar, ra) ** 2
    cost = xcost + yawcost + ucost
    return xcost


def compute_total_cost(start, end, cost):
    total_cost = []
    for i in range(len(start)):
        total_cost = np.append(total_cost, sum(cost[start[i]:end[i]]))
    return total_cost


def load_object(str):
    """
    Function to load to a pickle file
    :param str:
    :return:
    """
    with open(str, 'rb') as handle:
        return pickle.load(handle)


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def test_opt(Ac, bc, ts, waypt, order, num_steps, p, coeff, trained_model_state, value, aug_test_state):

    def calc_cost_GD(x0, coeff):
        coeff = coeff.reshape([4, 36])
        return linear_map_coeff(ts, order, num_steps, p, x0, trained_model_state, coeff, value)

    init = aug_test_state[0:4]

    print(coeff.shape)
    print(Ac(ts, order, coeff).shape)
    print(bc(waypt, order, ts, coeff).shape)

    A = Ac(ts, order, coeff)
    b = bc(waypt, order, ts, coeff)

    flat_coeff = np.reshape(coeff, [coeff.shape[0], -1])

    proj = onp.linalg.pinv(A) @ (b.T - A @ flat_coeff.T) + flat_coeff.T

    print("Proj shape", proj.shape)

    coeff = coeff.ravel()
    pg = ProjectedGradient(partial(calc_cost_GD, init), projection=projection_affine_set, maxiter=1)
    # solution = pg.run(coeff, hyperparams_proj=(proj.T, np.zeros([A.shape[0], 4])))
    solution = pg.run(coeff, hyperparams_proj=(proj.T.ravel(), 0.0))

    print("Solution", solution.params)

    prev_val = solution.state.error
    cur_sol = solution
    for j in range(50):
        next_sol = pg.update(cur_sol.params, cur_sol.state, hyperparams_proj=(Ac(ts, cur_sol.params), bc(waypt, ts, cur_sol.params)))
        val = next_sol.state.error
        if val < prev_val:
            solution = next_sol

        prev_val = val
        cur_sol = next_sol

    print(solution.state.error)

    ref = coeff2traj(solution.params, order, ts, num_steps)[1].T.flatten()

    return ref


def generate_polynomial_coeffs(start, end, T, order):
    """
    Generates a polynomial trajectory from start to end over time T
    start: start state
    end: end state
    T: total time
    order: order of the polynomial
    """
    # Define the time vector
    t = np.linspace(0, 1, T)

    # Solve for the polynomial coefficients
    coeffs = np.zeros(order + 1)
    coeffs = np.polyfit(t, t * (end - start) + start, order)

    # Evaluate the polynomial at the desired time steps
    polynomial = np.zeros(T)
    polynomial = np.polyval(coeffs[::-1], t)
    trajectory = polynomial + start

    return coeffs


def generate_polynomial_trajectory(start, end, T, order):
    """
    Generates a polynomial trajectory from start to end over time T
    start: start state
    end: end state
    T: total time
    order: order of the polynomial
    """
    # Define the time vector
    t = np.linspace(0, 1, T)

    # Solve for the polynomial coefficients
    coeffs = np.zeros(order + 1)
    coeffs = np.polyfit(t, t * (end - start) + start, order)

    # Evaluate the polynomial at the desired time steps
    polynomial = np.zeros(T)
    polynomial = np.polyval(coeffs[::-1], t)
    trajectory = polynomial + start

    return trajectory



def main():

    # Write a loop to iterate through different bags and compute costs and make plots
    #rhos = [0, 1, 5, 10, 20, 50, 100]
    rhos = [0]
    horizon = 300

    p = 4 + 4 * horizon
    q = 4

    with open(r"/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/params.yaml") as f:
        yaml_data = yaml.load(f, Loader=yaml.RoundTripLoader)

    num_hidden = yaml_data['num_hidden']
    batch_size = yaml_data['batch_size']
    learning_rate = yaml_data['learning_rate']

    # Load the trained model
    model = MLP(num_hidden=num_hidden, num_outputs=1)
    # Printing the model shows its attributes
    print(model)

    rng = jax.random.PRNGKey(427)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (batch_size, p))  # Batch size 64, input size p
    # Initialize the model
    params = model.init(init_rng, inp)

    optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)

    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                                params=params,
                                                tx=optimizer)

    rospy.init_node('inference_jax', anonymous=True)

    mav_id = 1

    # Creating MAV objects
    mav_namespace = 'dragonfly'

    mav_obj = KrMavInterface(mav_namespace, mav_id)

    mav_obj.motors_on()
    mav_obj.take_off()

    rospy.sleep(1)

    times_init = []
    times_opt = []
    times_poly = []
    duration = []

    for rho in rhos:

        # Coeff constraints
        #Ac = nonlinear._coeff_constr_A

        #bc = nonlinear._coeff_constr_b

        num_traj = 10
        np.random.seed(89)

        ref_traj = []

        dim = 4

        ref_coeff = np.zeros([num_traj, dim, horizon])


        for i in range(num_traj):
            # Testing on Lissajous curve

            x_amp = 0.65 * np.random.random()
            y_amp = 0.55 * np.random.random()
            z_amp = 0.55 * np.random.random()
            yaw_amp = 0.6414 * np.random.random()
            x_num_periods = 1
            y_num_periods = 1
            z_num_periods = 1
            yaw_num_periods = 1
            period = 3

            """# For using KR Tracker
            goal = LissajousTrackerGoal()
            # for i in range(10):
            goal.x_amp = x_amp
            goal.y_amp = y_amp
            goal.z_amp = z_amp
            goal.yaw_amp = yaw_amp
            goal.x_num_periods = x_num_periods
            goal.y_num_periods = y_num_periods
            goal.z_num_periods = z_num_periods
            goal.yaw_num_periods = yaw_num_periods
            goal.period = period
            goal.num_cycles = 1
            # goal.ramp_time = 2

            start = rospy.Time.now()
            times_kr.append(start)
            mav_obj.lissajous_tracker_client.send_goal(goal)
            success = mav_obj.transition_service_call('LissajousTracker')

            if not success:
                rospy.logwarn("Failed to transition to trajectory tracker (is there an active goal?)")

            rospy.logwarn("Waiting for traj to run")
            # mav_obj.traj_tracker_client.wait_for_result()
            mav_obj.lissajous_tracker_client.wait_for_result()
            end = rospy.Time.now()
            times_kr.append(end)
            duration.append((end - start).to_sec())
            print("Durations so far", duration)"""


            ref = generate_lissajous_traj(np.linspace(0, 3, 301), x_num_periods, y_num_periods, z_num_periods,
                                          yaw_num_periods, period, x_amp, y_amp, z_amp, yaw_amp)

            model_save = yaml_data['save_path'] + str(rho)

            trained_model_state = restore_checkpoint(model_state, model_save)

            mlp_t = load_torch_model(trained_model_state)

            print(mlp_t)

            vf = valuefunc.MLPValueFunc(mlp_t)

            vf.network = mlp_t

            dim = 4

            segments = 6
            ts = np.linspace(0, 3, segments + 1)
            order = 4
            # Tref = 300
            waypt = np.array(ref)[:, 0::50]
            # waypt[:, 3] = waypt[:, 3] + 0.3
            offset = min(waypt[2, :])
            waypt[2, :] = waypt[2, :] - offset + 1
            num_steps = 300

            mav_obj.send_wp_block(waypt[0, 0], waypt[1, 0], waypt[2, 0], 0)

            #rospy.sleep(1)

            # ref_coeff, init_coeff = quadratic.generate(waypt.T, ts, order, horizon, dim, None, 0)

            for k in range(dim):
                for j in range(segments):
                    ref_coeff[i, k, j*50:(j+1)*50] = generate_polynomial_trajectory(waypt[k, j], waypt[k, j + 1],
                                                                                            50, order)

            """init_coeff = np.zeros([dim, segments, order+1])

            for k in range(dim):
                for j in range(segments):
                    init_coeff[k, j, :] = generate_polynomial_coeffs(waypt[k, j], waypt[k, j+1], 50, order)

            print(init_coeff)

            pos, vel, acc, jerk, yaw, yaw_dot = compute_pos_vel_acc(horizon, init_coeff, segments, ts)"""

            """pos = ref_coeff[i, 0:3, :].reshape((3, horizon))
            #yaw = ref_coeff[i, 3, :].reshape(horizon)
            yaw = np.zeros(horizon)
            vel = np.zeros([dim, horizon])
            acc = np.zeros([dim, horizon])
            jerk = np.zeros([dim, horizon])
            yaw_dot = np.zeros(horizon)"""
            vel = np.zeros([dim, horizon])
            acc = np.zeros([dim, horizon])
            jerk = np.zeros([dim, horizon])

            pos = ref_coeff[i, 0:3, :].reshape((3, horizon))
            yaw = ref_coeff[i, 3, :].reshape(horizon)
            yaw_dot = np.zeros(horizon)

            success = mav_obj.transition_service_call('NullTracker')
            if not success:
                rospy.logwarn("Failed to transition to null tracker (is there an active goal?)")

            rospy.logwarn("Waiting for traj to run")

            vel[:, horizon - 1] = 0
            acc[:, horizon - 1] = 0
            jerk[:, horizon - 1] = 0
            yaw[horizon - 1] = 0
            yaw_dot[horizon - 1] = 0

            start = rospy.Time.now()
            times_poly.append(start)
            rate = rospy.Rate(100)
            for j in range(horizon):
                # Pass commands to publisher at a certain frequency
                #mav_obj.publish_pos_cmd(pos[:, j], vel[:, j], acc[:, j], jerk[:, j], yaw[j], yaw_dot[j])
                mav_obj.publish_pos_cmd(pos[0, j], pos[1, j], pos[2, j], vel[0, j], vel[1, j], vel[2, j], acc[0, j],
                                        acc[1, j], acc[2, j], jerk[0, j], jerk[1, j], jerk[2, j], yaw[j], yaw_dot[j])
                rate.sleep()
            end = rospy.Time.now()
            times_poly.append(end)
            duration.append((end - start).to_sec())
            print("Durations so far", duration)

            # Send waypoint blocking
            mav_obj.send_wp_block(pos[0, horizon - 1], pos[1, horizon - 1], pos[2, horizon - 1], 0.0, 0, 0,
                                  False) # x, y, z, yaw, vel, acc, relative

            # nn_coeff = quadrotor.generate(torch.tensor(waypt), ts, order, horizon, dim, rho, vf,
            #                              torch.zeros([4, 6, 6]).double(), num_iter=20, lr=0.005)
            # order = 2
            """nn_coeff = quadrotor.generate(torch.tensor(waypt), ts, order, horizon, dim, rho, vf, torch.tensor(init_coeff), num_iter=100, lr=0.001)


            pos, vel, acc, jerk, yaw, yaw_dot = compute_pos_vel_acc(horizon, nn_coeff, segments, ts)

            poly_pos = compute_pos_vel_acc(horizon, init_coeff, segments, ts)
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')
            # ttraj = actual_traj.copy()
            # axes = plt.gca(projection='3d')
            print(pos.shape)
            axes.plot3D(pos[0, :], pos[1, :], pos[2, :])
            axes.plot3D(poly_pos[0][0, :], poly_pos[0][1, :], poly_pos[0][2, :])
            # axes.plot3D(ref_coeff[:, 0], ref_coeff[:, 2], ref_coeff[:, 3])
            axes.set_xlim(-1, 1)
            axes.set_zlim(0, 4)
            axes.set_ylim(-1, 1)
            axes.plot3D(waypt[0, :], waypt[1, :], waypt[2, :], '*')

            plt.show()

            success = mav_obj.transition_service_call('NullTracker')
            if not success:
                rospy.logwarn("Failed to transition to null tracker (is there an active goal?)")

            rospy.logwarn("Waiting for traj to run")

            vel[:, horizon - 1] = 0
            acc[:, horizon - 1] = 0
            jerk[:, horizon - 1] = 0
            yaw[horizon - 1] = 0
            yaw_dot[horizon - 1] = 0

            start = rospy.Time.now()
            times_opt.append(start)
            rate = rospy.Rate(100)
            for j in range(horizon):
                # Pass commands to publisher at a certain frequency
                #mav_obj.publish_pos_cmd(pos[:, j], vel[:, j], acc[:, j], jerk[:, j], yaw[j], yaw_dot[j])
                mav_obj.publish_pos_cmd(pos[0, j], pos[1, j], pos[2, j], vel[0, j], vel[1, j], vel[2, j], acc[0, j],
                                    acc[1, j], acc[2, j], jerk[0, j], jerk[1, j], jerk[2, j], yaw[j], yaw_dot[j])
                rate.sleep()
            end = rospy.Time.now()
            times_opt.append(end)
            duration.append((end - start).to_sec())
            print("Durations so far", duration)

            # Send waypoint blocking
            mav_obj.send_wp_block(pos[0, horizon - 1], pos[1, horizon - 1], pos[2, horizon - 1], 0.0, 0, 0,
                                  False)"""  # x, y, z, yaw, vel, acc, relative

            # Save initial references to a file

            # aug_state = np.append(ref_traj[0:4], ref_traj)

            #value = 1
            #opt_ref.append(test_opt(Ac, bc, ts, waypt, order, num_steps, dim, onp.array(init_coeff.reshape([4, (order + 1) * segments], order='F')), trained_model_state, value, aug_state))
                                    #onp.array(init_coeff.reshape([init_coeff.shape[0], -1])), trained_model_state, value, aug_state))

            """from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')
            # ttraj = actual_traj.copy()
            # axes = plt.gca(projection='3d')
            print(pos.shape)
            axes.plot3D(pos[0, :], pos[1, :], pos[2, :])
            #axes.plot3D(ref_coeff[:, 0], ref_coeff[:, 2], ref_coeff[:, 3])
            axes.set_xlim(-1, 1)
            axes.set_zlim(0, 4)
            axes.set_ylim(-1, 1)
            axes.plot3D(waypt[0, :], waypt[1, :], waypt[2, :], '*')

            plt.show()"""

    save_object(times_poly, "/home/anusha/poly_inf_times.pkl")

if __name__ == '__main__':
    main()


