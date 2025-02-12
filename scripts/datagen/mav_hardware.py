#! /usr/bin/env python3
import matplotlib.pyplot as plt
import rospy
import numpy as np
import random
import time

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool, Trigger
from kr_tracker_msgs.msg import TrajectoryTrackerAction, TrajectoryTrackerGoal

from layered_ref_control.mav_layer_interface import KrMavInterface
from trajgen import nonlinear, quadratic, trajutils, quadrotor
from learning import lstd, trajdata, valuefunc
from env import baseenv, linearenv, controller
import torch
import pickle
import sys


PI = np.pi


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


def generate_waypoints(segments):
    """
    Function to generate a list of waypoints given the number of waypoints as an argument
    :return:
    """
    """sample = list()
    goals = np.abs(np.random.randn(3, 2))
    points = np.linspace(0, 1, segments + 1) * 0.6 + 0.2
    for i in range(segments + 1):
        sample.append(np.abs(goals[:, 0] + (goals[:, 1] - goals[:, 0]) * points[i]))
    idx = range(3, 10)
    waypt = np.zeros([segments + 1, 10])
    waypt[:, idx] = None
    waypt[:, 0:3] = np.vstack(sample)
    return waypt"""
    wp = list()
    # wp.append([0, 0, 2] + np.abs(np.random.randn(3)))
    wp.append([0, 0, 2])
    wp.append([0, 0.1, 2] + np.abs(np.random.random(3)))
    wp.append([0.2, 0.2, 2]) #+ np.abs(np.random.randn(3)))
    wp.append([0.3, 0.3, 2])
    wp.append([0.3, 0, 1.5])
    wp.append([0.3, 0.3, 1.5])
    wp.append([0, 0.3, 1.5]) #+ np.abs(np.random.randn(3)))
    # idx = range(3, 10)
    waypt = np.zeros([segments + 1, 4])
    waypt[:, 3] = 0
    waypt[:, 0:3] = np.vstack(wp)
    return waypt


def reference_planner(segments, order, ts, Tref, p, rho, vf):
    """

    :return:
    """
    # Generate a waypoints and the min-jerk trajectory
    waypoints = generate_waypoints(segments)
    _, minjerk_coeffs = quadratic.generate(waypoints, ts, order, Tref, p, None, 0)
    # Generate regularized trajectory from gradient descent on NN
    """waypt = np.zeros([segments + 1, 10])
    idx = range(3, 10)
    waypt[:, 4:] = 0
    waypt[:, 0:4] = waypoints
    coeffs = np.zeros([10, segments, order + 1])
    # idx = range(3, 10)
    coeffs[0:4, :, :] = minjerk_coeffs
    coeffs[4:, :, :] = 0
    p_dim = 10
    nn_coeff = nonlinear.generate(torch.tensor(waypt.T), ts, order, Tref, p_dim, rho, vf, torch.tensor(coeffs),
                                  num_iter=100, lr=0.01)"""
    return minjerk_coeffs, waypoints


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

    print("Pos array shape", np.array(pos).shape)

    return np.vstack(pos), np.vstack(vel), np.vstack(acc), np.vstack(jerk), yaw_ref, yawdot_ref


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def main():

    # Accept options from commandline
    # print(sys.argv[0])
    #rho = float(sys.argv[1])
    #print(rho)
    # time_file = sys.argv[1]
    # traj_file = sys.argv[2]

    """if traj_file == 'kr':
        print("Need to run KR Trajectory Tracker")
    else:
        with open(traj_file, 'rb') as handle:
            nn = pickle.load(handle)"""

    rospy.init_node('mav_hardware', anonymous=True)

    # zero_time = rospy.Time()
    duration = []
    times = []

    sim = False
    mav_id = 1

    if not sim:
        mav_id = 24

    # Creating MAV objects
    mav_namespace = 'dragonfly'

    mav_obj = KrMavInterface(mav_namespace, mav_id)
    rospy.sleep(1)

    if sim:
        # Motor On / Take Off
        # if i == 0:
        mav_obj.motors_on()
        mav_obj.take_off()

    seed = [np.random.seed(10), np.random.seed(11), np.random.seed(12), np.random.seed(15), np.random.seed(17), np.random.seed(28), np.random.seed(29)]

    for i in range(7):

        np.random.seed(seed[i])
        if traj_file == 'kr':
            x_amp = 0.65 * np.random.random()
            y_amp = 0.65 * np.random.random()
            z_amp = 0.65 * np.random.random()
            yaw_amp = 0.3414 * np.random.random()
            x_num_periods = 1
            y_num_periods = 1
            z_num_periods = 1
            yaw_num_periods = 1
            period = 3  # + np.random.random()

            ref = generate_lissajous_traj(np.linspace(0, 3, 301), x_num_periods, y_num_periods, z_num_periods,
                                          yaw_num_periods,
                                          period, x_amp,
                                          y_amp, z_amp, yaw_amp)

            p = 4

            segments = 6
            ts = np.linspace(0, 3, segments + 1)
            order = 5
            Tref = 300
            waypt = np.array(ref)[:, 0::50]
            offset = min(waypt[2, :])
            # print("Negative offset", offset)
            waypt[2, :] = waypt[2, :] - offset + 1.3

            print("Time steps", ts)

            goal = TrajectoryTrackerGoal()
            for i in range(1, segments+1):
                wp = Pose()
                wp.position.x = waypt[0, i]
                wp.position.y = waypt[1, i]
                wp.position.z = waypt[2, i]
                goal.waypoints.append(wp)
                goal.waypoint_times.append(ts[i])


            # rospy.sleep(1)
            mav_obj.send_wp_block(waypt[0, 0], waypt[1, 0], waypt[2, 0], 0)
            start = rospy.Time.now()
            times.append(start)
            mav_obj.traj_tracker_client.send_goal(goal)
            success = mav_obj.transition_service_call('TrajectoryTracker')
            if not success:
                rospy.logwarn("Failed to transition to trajectory tracker (is there an active goal?)")

            rospy.logwarn("Waiting for traj to run")
            mav_obj.traj_tracker_client.wait_for_result()
            end = rospy.Time.now()
            times.append(end)
            duration.append((end - start).to_sec())
            print("Durations so far", duration)


        else:
            pos = nn[i][:, 0:3].T
            vel = nn[i][:, 3:6].T
            acc = nn[i][:, 6:9].T
            jerk = nn[i][:, 9:12].T
            yaw = nn[i][:, 12]
            yaw_dot = nn[i][:, 13]
            # pos, vel, acc, jerk, yaw, yaw_dot = nn[i]

            # offset = min(pos[pos < 0])
            # pos = pos - offset + 0.5
            #rospy.sleep(1)
            #pos[2, :] = pos[2, :] + 0.3

            mav_obj.send_wp_block(pos[0, 0], pos[1, 0], pos[2, 0], 0)

            # rospy.sleep(1)

            Tref = 300
            fig, ax = plt.subplots(1, 4)
            ax[0].plot(range(0, Tref), pos[:3, :].T, label=['x', 'y', 'z'])
            ax[1].plot(range(0, Tref), vel[:3, :].T, label=['vx', 'vy', 'vz'])
            ax[2].plot(range(0, Tref), acc[:3, :].T, label=['ax', 'ay', 'az'])
            ax[3].plot(range(0, Tref), jerk[:3, :].T, label=['jx', 'jy', 'jz'])
            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
            ax[3].legend()
            # plt.legend(handles=['position', 'velocity', 'acceleration', 'jerk'])
            # plt.show()
            plt.savefig('/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/plots/traj_inf' + str(i) + '.png')

            success = mav_obj.transition_service_call('NullTracker')
            if not success:
                rospy.logwarn("Failed to transition to null tracker (is there an active goal?)")

            rospy.logwarn("Waiting for traj to run")

            vel[:, Tref - 1] = 0
            acc[:, Tref - 1] = 0
            jerk[:, Tref - 1] = 0
            yaw[Tref - 1] = 0
            yaw_dot[Tref - 1] = 0

            start = rospy.Time.now()
            times.append(start)
            rate = rospy.Rate(100)
            for i in range(Tref):
                # Pass commands to publisher at a certain frequency
                mav_obj.publish_pos_cmd(pos[:, i], vel[:, i], acc[:, i], jerk[:, i], yaw[i], yaw_dot[i])
                rate.sleep()
            end = rospy.Time.now()
            times.append(end)
            duration.append((end - start).to_sec())
            print("Durations so far", duration)
            # rospy.sleep(1)
            # Send waypoint blocking
            mav_obj.send_wp_block(pos[0, Tref-1], pos[1, Tref-1], pos[2, Tref-1], 0.0, 0, 0, False)  # x, y, z, yaw, vel, acc, relative
            rospy.sleep(1)
        # del mav_obj
        # print("Time duration so far", now)
        # times.append(now)

    # sample usage
    #save_object(duration,
    #            '/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/duration.pkl')
    # save_object(times,
    #             '/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/times.pkl')
    save_object(times, time_file)
    # np.savetxt('./durations', duration)




if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    # finally:
    #    rosservice call /rosbag_record_service/stop
