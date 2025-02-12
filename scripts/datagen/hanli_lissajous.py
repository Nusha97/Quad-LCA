#! /usr/bin/env python3

"""
Generate training data:
Sample waypoints from a Lissajous, squared, or heart-shaped curve
Simplely replan the trajectory for longer time period with fixed waypoints
"""

# from os import times
import matplotlib.pyplot as plt
import rospy
import numpy as np
import jax.numpy as jnp
import random

import ruamel.yaml as yaml
from flax.training import train_state
import optax
import jax
from mlp_jax import MLP
from model_learning import restore_checkpoint

from kr_tracker_msgs.msg import LissajousTrackerGoal, LissajousTrackerAction
from layered_ref_control.mav_layer_interface import KrMavInterface

from trajgen import quadratic, nonlinear_jax, nonlinear_flax, check_jax
# from trajgen import nonlinear, quadratic, trajutils, quadrotor
# from learning import lstd, trajdata, valuefunc
from env import baseenv, linearenv, controller
# import torch
import pickle
import sys

PI = np.pi


def generate_lissajous_traj(s, x_num_periods, y_num_periods, z_num_periods, yaw_num_periods, period, x_amp, y_amp,
                            z_amp, yaw_amp):
    """
    Function to generate Lissajous trajectory
    :return:
    """
    x = lambda a: x_amp * (1 - np.cos(2 * PI * x_num_periods * a / period))
    y = lambda a: y_amp * np.sin(2 * PI * y_num_periods * a / period)
    z = lambda a: z_amp * np.sin(2 * PI * z_num_periods * a / period)
    yaw = lambda a: yaw_amp * np.sin(2 * PI * yaw_num_periods * a / period)
    return np.array([x(s), y(s), z(s), yaw(s)])


def generate_square_traj(t, side_length):
    """
    Function to generate square trajectory in x and z
    """
    # Calculate the number of time samples
    num_samples = len(t)

    # Calculate the x and z coordinates of the square trajectory
    x = np.concatenate([
        np.linspace(-side_length / 2, side_length / 2, num_samples // 4),
        np.full(num_samples // 4, side_length / 2),
        np.linspace(side_length / 2, -side_length / 2, num_samples // 4),
        np.full(num_samples // 4, -side_length / 2)
    ])
    # print("len(x): ", len(x))

    z = np.concatenate([
        np.full(num_samples // 4, -side_length / 2),
        np.linspace(-side_length / 2, side_length / 2, num_samples // 4),
        np.full(num_samples // 4, side_length / 2),
        np.linspace(side_length / 2, -side_length / 2, num_samples // 4)
    ])

    # z = np.concatenate([
    #     np.full(num_samples // 2, -side_length / 2),
    #     np.full(num_samples // 2, side_length / 2)
    # ])
    print("z: ", z)

    # Set the y and yaw coordinates to zeros (assuming a flat trajectory), size as x and z
    y = np.zeros_like(x)
    # print("len(y): ", len(y))
    yaw = np.zeros_like(x)
    # print("len(yaw): ", len(yaw))
    # y = np.zeros_like(t)
    # print("len(y): ", len(y))
    # yaw = np.zeros_like(t)
    # print("len(yaw): ", len(yaw))

    # Combine the coordinates into a single array
    traj = np.vstack((x, y, z, yaw))

    return traj


def generate_heart_traj(t, scale):
    # Calculate the number of time samples
    num_samples = len(t)

    # Calculate the theta parameter
    theta = np.linspace(0, 2 * np.pi, num_samples)

    # Calculate the x and z coordinates of the heart trajectory
    x = scale * 16 * np.sin(theta) ** 3
    z = scale * (13 * np.cos(theta) - 5 * np.cos(2 * theta) - 2 * np.cos(3 * theta) - np.cos(4 * theta))

    # Set the y and yaw coordinates to zeros (assuming a flat trajectory)
    y = np.zeros_like(t)
    yaw = np.zeros_like(t)

    # Combine the coordinates into a single array
    traj = np.vstack((x, y, z, yaw))

    return traj


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
            coeff_new[i, j + 1:] = t.coefficients
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
        ref.append(poly[k](tt - ts[k]))
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

    return np.vstack(pos), np.vstack(vel), np.vstack(acc), np.vstack(jerk), yaw_ref, yawdot_ref


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
    #coeffs = np.zeros(order + 1)
    coeffs = np.polyfit(t, t * (end - start) + start, order)

    # Evaluate the polynomial at the desired time steps
    #polynomial = np.zeros(T)
    polynomial = np.polyval(coeffs[::-1], t)
    trajectory = polynomial + start

    return coeffs


def replan_trajectory(prev_waypoints, new_waypoints, duration, num_waypoints_per_segment, order, p,
                      current_waypoint_index):
    # TO DO: traj's continuity & generate new waypoints?

    start = rospy.Time.now()

    # Generate time samples for the new trajectory
    ts = np.linspace(0, duration, num_waypoints_per_segment)

    if current_waypoint_index == 0:
        # Initial waypoints
        waypoints = prev_waypoints
    else:
        # Calculate the number of waypoints for the new trajectory
        # num_prev_waypoints = prev_waypoints.shape[0]
        num_new_waypoints = new_waypoints.shape[0]
        num_counted_prev_waypoints = num_waypoints_per_segment - num_new_waypoints
        # print("num_prev_waypoints: ", num_prev_waypoints)
        # print("num_new_waypoints: ", num_next_waypoints)

        # Determine the start index for the previous waypoints to be included
        start_idx = current_waypoint_index - num_counted_prev_waypoints
        print("start_idx: ", start_idx)

        # Get the subset of previous waypoints to be combined with the next waypoints
        selected_prev_waypoints = prev_waypoints[start_idx:current_waypoint_index]

        # Concatenate the selected previous waypoints with the next waypoints
        waypoints = np.vstack((selected_prev_waypoints, new_waypoints))
        print("selected_previous_wp's shape: ", selected_prev_waypoints.shape)
        print("new_up's shape: ", new_waypoints.shape)

    print("next interation's waypoints' shape: ", waypoints.shape)

    # Generate the new trajectory using the concatenated waypoints
    _, new_traj_coeffs = quadratic.generate(waypoints, ts, order, duration * 100, p, None, 0)
    # print("new_traj_coeffs' shape: ", new_traj_coeffs.shape)
    # print("new_traj_coeffs: ", new_traj_coeffs)

    # Set the current waypoint index for the next iteration
    current_waypoint_index += num_waypoints_per_segment

    end = rospy.Time.now()

    generation_time = end - start

    print("generation time: ", generation_time)

    return waypoints, new_traj_coeffs, current_waypoint_index


def simple_replan(selected_waypoints, vf, duration, order, p, idx=None):
    """
    Function to generate a new trajectory using the selected waypoints
    """
    start = rospy.Time.now()

    # Generate time samples for the new trajectory
    ts = np.linspace(0, duration, selected_waypoints.shape[0])

    new_traj_coeffs = np.zeros([p, len(selected_waypoints) - 1, order + 1])

    for k in range(p):
        for j in range(len(selected_waypoints) - 1):
            new_traj_coeffs[k, j, :] = generate_polynomial_trajectory(selected_waypoints.T[k, j],
                                                                      selected_waypoints.T[k, j + 1], duration * 100+1, order)

    # Generate the new trajectory using the concatenated waypoints
    _, min_jerk_coeffs = quadratic.generate(selected_waypoints, ts, order, duration * 100+1, p, None, 0)
    #print("new_traj_coeffs' shape: ", min_jerk_coeffs.shape)
    # print("new_traj_coeffs: ", new_traj_coeffs)


    #nn_coeffs = nonlinear_jax.generate(selected_waypoints.T, ts, order, duration * 100, p, rho, vf, min_jerk_coeffs,
    #                                   num_iter=100, lr=0.0001)

    #nn_coeffs = check_jax.generate(selected_waypoints.T, ts, order, duration * 100, p, vf, jnp.array(min_jerk_coeffs))
    nn_coeffs, pred = nonlinear_flax.modify_reference(selected_waypoints, ts, duration * 100, order, p, vf, new_traj_coeffs)
    #nn_coeffs = load_object(
    #    r"/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/nn_coeffs" + str(
    #        rho) + ".pkl")

    end = rospy.Time.now()

    generation_time = end - start

    print("generation time: ", generation_time)

    #return min_jerk_coeffs
    return nn_coeffs
    #return new_traj_coeffs


def save_object(obj, filename):
    """
    Function to save to a pickle file
    :param obj:
    :param filename:
    :return:
    """
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(str):
    """
    Function to load to a pickle file
    :param str:
    :return:
    """
    with open(str, 'rb') as handle:
        return pickle.load(handle)


def main():
    mav_name = 1
    # 0512
    print("Mav name ", mav_name)
    rospy.init_node('lissajous_data', anonymous=True)

    # Creating MAV objects
    mav_namespace = 'dragonfly'
    # duration = []
    times = []

    # Create the controller object
    mav_obj = KrMavInterface(mav_namespace, mav_name)
    rospy.sleep(3)

    mav_obj.motors_on()
    mav_obj.take_off()

    rospy.sleep(3)


    # parameters for lissajous trajectory
    x_amp = 2
    y_amp = 2
    z_amp = 0.8
    yaw_amp = 0.2

    x_num_periods = 2
    y_num_periods = 2
    z_num_periods = 2
    yaw_num_periods = 2

    # total period for all the trajectories
    period = 6
    p = 4
    order = 7
    Tref = period * 100

    movig_widow = 3
    num_waypoints_per_segment = 5
    duration = 3  # Duration of each replanning iteration

    rho = 1
    with open(r"/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/params.yaml") as f:
        yaml_data = yaml.load(f, Loader=yaml.RoundTripLoader)

    num_hidden = yaml_data['num_hidden']
    batch_size = yaml_data['batch_size']
    learning_rate = yaml_data['learning_rate']
    # Load the trained model
    model = MLP(num_hidden=num_hidden, num_outputs=1)
    rng = jax.random.PRNGKey(427)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (batch_size, p * duration * 100 + p))  # Batch size 64, input size p
    # Initialize the model
    params = model.init(init_rng, inp)
    optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)
    #optimizer = optax.adam(learning_rate=learning_rate)
    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                                params=params,
                                                tx=optimizer)
    model_save = yaml_data['save_path'] + str(rho)
    print(model_save)
    trained_model_state = restore_checkpoint(model_state, model_save)
    vf = model.bind(trained_model_state.params)

    # Time samples
    t = np.linspace(0, period, period * 100 + 1)

    # -------------------------------------------------------
    # # Generate the waypoints for the entire lissajous trajectory
    ref = generate_lissajous_traj(np.linspace(0, period, period*100+1), x_num_periods, y_num_periods, z_num_periods,
                                   yaw_num_periods, period, x_amp, y_amp, z_amp, yaw_amp)

    # -------------------------------------------------------

    # -------------------------------------------------------
    # # generate the waypoints for the entire square trajectory
    # side_length = 2.0  # Length of each side of the square

    # ref = generate_square_traj(t, side_length)
    # -------------------------------------------------------

    # -------------------------------------------------------
    # generate the waypoints for the entire heart-shaped trajectory
    #scale = 0.1
    #ref = generate_heart_traj(t, scale)
    # -------------------------------------------------------
    rospy.logwarn("ref shape: {}".format(ref.shape))

    waypt = np.array(ref)[:, 0::30]
    # waypt = np.array(ref)[:, 0::50]
    # Get the number of segments and time samples
    segments = len(waypt.T) - 1
    print("Segments:", segments)
    ts = np.linspace(0, period, segments + 1)

    offset = min(waypt[2, :])
    print("Negative offset", offset)
    waypt[2, :] = waypt[2, :] - offset + 1
    mav_obj.publish_waypoints(waypt, 1.0, 0.0, 0.0, 0.9)
    ref[2, :] = ref[2, :] - offset + 1
    mav_obj.publish_ref_traj(ref)

    # we don't need to offset the z axis in sim
    #offset = min(waypt[2, :])
    #print("Negative offset", offset)
    #waypt[2, :] = waypt[2, :] - offset + 0.3

    # do the offset for ref and publish
    #ref[2, :] = ref[2, :] - offset + 1
    wp = waypt[:, :12].T
    _, min_jerk_coeffs = quadratic.generate(wp, np.linspace(0, 9, len(wp)), order, 9 * 100, p, None, 0)


    new_traj_coeffs = np.zeros([p, len(wp) - 1, order + 1])

    for k in range(p):
        for j in range(len(wp) - 1):
            new_traj_coeffs[k, j, :] = generate_polynomial_trajectory(wp.T[k, j], wp.T[k, j + 1], 900, order)
    #pos, vel, acc, jerk, yaw, yaw_dot = compute_pos_vel_acc(9 * 100, new_traj_coeffs, len(wp)-1, np.linspace(0, 9, len(wp)))
    pos, vel, acc, jerk, yaw, yaw_dot = compute_pos_vel_acc(9 * 100, min_jerk_coeffs, len(wp) - 1, np.linspace(0, 9, len(wp)))

    #mav_obj.publish_ref_traj(pos)



    # print("Waypt shape", waypt.shape)   #(4,11)
    # print("Waypoints", waypt[:, 0])
    # print("all waypoints", waypt)
    # 0515
    # publish all the waypts
    mav_obj.publish_waypoints(waypt, 0.0, 0.0, 1.0, 0.5)
    print(len(waypt.T))

    #rospy.sleep(10)

    # when to call nulltracker?
    success = mav_obj.transition_service_call('NullTracker')
    # success = mav_obj.transition_service_call('TrajectoryTracker')
    if not success:
        rospy.logwarn("Failed to transition to null tracker (is there an active goal?)")

    rospy.logwarn("Waiting for traj to run")

    # Main loop
    rate = rospy.Rate(100)

    # # Send waypoint blocking for the first waypoint
    # mav_obj.send_wp_block(waypt[0, 0], waypt[1, 0], waypt[2, 0], 0.0, 0, 0, False)  # x, y, z, yaw, vel, acc, relative

    # # sleep for 0.5 second to make sure the trajectory is running
    # rospy.sleep(0.5)

    start_ = rospy.Time.now()
    times.append(start_)

    # Initialize the current waypoint indexcc
    current_waypoint_index = 0
    idx = 0
    while current_waypoint_index < len(waypt.T) - num_waypoints_per_segment + 2:
        print("current_waypoint_index", current_waypoint_index)
        start_iteration = rospy.Time.now()
        # Determine the start and end indices of the next waypoints and trajectory to consider
        start_idx = current_waypoint_index
        end_idx = start_idx + num_waypoints_per_segment
        # print("start_idx", start_idx)
        # print("end_idx", end_idx)

        # Select the waypoints for replanning
        selected_waypoints = waypt.T[start_idx:end_idx]
        print("selected_waypoints's shape", selected_waypoints.shape)
        # mav_obj.publish_waypoints(selected_waypoints.T, 1.0, 0.0, 0.0, 0.9)

        # Replan the trajectory based on previous and next waypoints
        start_replan = rospy.Time.now()
        new_traj_coeffs = simple_replan(selected_waypoints, vf, duration, order, p, idx)
        idx += 1
        end_replan = rospy.Time.now()
        # print("Replanning time", end_replan - start_replan)

        # Update the current waypoint index
        current_waypoint_index += movig_widow

        print("current_waypoint_index after update", current_waypoint_index)

        # Compute position, velocity, acceleration, jerk, yaw, and yaw rate from the new trajectory
        start_compute = rospy.Time.now()
        segment_new = len(selected_waypoints) - 1
        print("segment_new", segment_new)
        ts_new = np.linspace(0, duration, segment_new + 1)
        print("ts_new", ts_new)
        pos, vel, acc, jerk, yaw, yaw_dot = compute_pos_vel_acc(duration * 100, new_traj_coeffs, segment_new, ts_new)

        # publish path
        mav_obj.publish_replan(pos)

        #vel[:, -1] = np.zeros(3)
        #acc[:, -1] = np.zeros(3)
        #jerk[:, -1] = np.zeros(3)  # if neccessary, set the last jerk to zero

        end_compute = rospy.Time.now()
        # print("Compute time", end_compute - start_compute)

        # Pass commands to the controller at a certain frequency
        print("len(pos.T)", len(pos.T))

        # fix the moving_window and num_waypoints_per_segment, publish pos_cmd from the second segment after the first iteration
        if current_waypoint_index == movig_widow:
            for i in range(len(pos.T)):
                mav_obj.publish_pos_cmd(pos[0, i], pos[1, i], pos[2, i], vel[0, i], vel[1, i], vel[2, i], acc[0, i],
                                        acc[1, i], acc[2, i], jerk[0, i], jerk[1, i], jerk[2, i], yaw[i], yaw_dot[i])
                rate.sleep()

        else:
            for i in range(len(pos.T) // segment_new, len(pos.T)):
                # rospy.logwarn("Publishing pos: %s", pos[:, i])
                # mav_obj.publish_pos_cmd(pos[:, i], vel[:, i], acc[:, i], jerk[:, i], yaw[i], yaw_dot[i])

                # 0530
                mav_obj.publish_pos_cmd(pos[0, i], pos[1, i], pos[2, i], vel[0, i], vel[1, i], vel[2, i], acc[0, i],
                                        acc[1, i], acc[2, i], jerk[0, i], jerk[1, i], jerk[2, i], yaw[i], yaw_dot[i])
                rate.sleep()

        end_iteration = rospy.Time.now()
        rospy.logwarn("Iteration time: %s", end_iteration - start_iteration)
        # start = rospy.Time.now()
        # # times.append(start)
        # mav_obj.send_wp_block(selected_waypoints[-1, 0], selected_waypoints[-1, 1], selected_waypoints[-1, 2], 0)
        # pos, vel, acc, jerk, yaw, yaw_dot = compute_pos_vel_acc(Tref, minjerk_coeffs, segments, ts)

        # print("pos values", pos)

        rospy.logwarn("Reached initial waypoints")
        # offset = min(pos[pos < 0])
        # pos = pos - offset + 0.5

        """from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        # ttraj = actual_traj.copy()
        # axes = plt.gca(projection='3d')
        # 0510
        mav_id = 1
        axes.plot3D(pos[0, :], pos[1, :], pos[2, :])
        axes.set_xlim(-1, 1)
        axes.set_zlim(0, 4)
        axes.set_ylim(-1, 1)
        axes.plot3D(waypt[0, :], waypt[1, :], waypt[2, :], '*')

        fig, ax = plt.subplots(1, 4)
        ax[0].plot(range(0, Tref), pos[:3, :].T, label=['x', 'y', 'z'])
        ax[1].plot(range(0, Tref), vel[:3, :].T, label=['vx', 'vy', 'vz'])
        ax[2].plot(range(0, Tref), acc[:3, :].T, label=['ax', 'ay', 'az'])
        ax[3].plot(range(0, Tref), jerk[:3, :].T, label=['jx', 'jy', 'jz'])
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[3].legend()
        # plt.savefig('./layered_ref_control/src/layered_ref_control/plots/traj_infot3'+str(mav_id)+'.png')
        # plt.savefig('./src/layered_ref_control/plots/traj_infot3'+str(mav_id)+'.png')

        # plt.legend(handles=['position', 'velocity', 'acceleration', 'jerk'])
        plt.show()
        # 0510
        """

    end_ = rospy.Time.now()
    times.append(end_)

    # send up block
    rospy.sleep(1)
    rospy.logwarn("Reached final waypoints and sending up block")
    mav_obj.send_wp_block(selected_waypoints[-1, 0], selected_waypoints[-1, 1], selected_waypoints[-1, 2], 0)

    # Land / Motors off
    mav_obj.land()
    rospy.sleep(3)
    mav_obj.motors_off()

    # save_object(times, "/home/mrsl_guest/times.pkl")

    # vel[:, Tref-1] = 0
    # acc[:, Tref-1] = 0
    # jerk[:, Tref-1] = 0
    # yaw[Tref-1] = 0
    # yaw_dot[Tref-1] = 0

    # rate = rospy.Rate(100)
    # for i in range(Tref):
    #     # Pass commands to publisher at a certain frequency
    #     mav_obj.publish_pos_cmd(pos[:, i], vel[:, i], acc[:, i], jerk[:, i], yaw[i], yaw_dot[i])
    #     rate.sleep()

    # # Send waypoint blocking
    # mav_obj.send_wp_block(pos[0, Tref-1], pos[1, Tref-1], pos[2, Tref-1], 0.0, 0, 0, False)  # x, y, z, yaw, vel, acc, relative
    # end = rospy.Time.now()
    # # times.append(end)
    # duration.append((end - start).to_sec())
    # print("Durations so far", duration)

    # save_object(duration, '/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/net_duration.pkl')
    # save_object(times,
    #            '/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/net_times.pkl')


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass