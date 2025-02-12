import pickle

import rosbag
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tf
from helper_functions import compute_cum_tracking_cost


def load_bag(str):
    """

    :param str:
    :return:
    """
    bag = rosbag.Bag(str)
    return bag


def rotationMatrixToQuaternion1(m):
    #q0 = qw
    t = np.matrix.trace(m)
    q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if(t > 0):
        t = np.sqrt(t + 1)
        q[3] = 0.5 * t
        t = 0.5/t
        q[0] = (m[2, 1] - m[1, 2]) * t
        q[1] = (m[0, 2] - m[2, 0]) * t
        q[2] = (m[1, 0] - m[0, 1]) * t

    else:
        i = 0
        if (m[1, 1] > m[0, 0]):
            i = 1
        if (m[2, 2] > m[i, i]):
            i = 2
        j = (i+1)%3
        k = (j+1)%3

        t = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (m[k, j] - m[j, k]) * t
        q[j] = (m[j, i] + m[i, j]) * t
        q[k] = (m[k, i] + m[i, k]) * t

    return q


def get_T(acc, g=9.81):
	"""
	Function to compute the intermediate T vector
	:param acc:
	:param g:
	:return:
	"""
	return [np.array([x[0], x[1], x[2]+g]) for x in acc]


def get_yc(yaw):
	"""
    Function to compute intermediate yc vector
	:return:
	"""
	Tref = len(yaw)
	temp = np.stack([-np.sin(yaw), np.cos(yaw), np.zeros(Tref)]).flatten()
	temp = temp.reshape((3, Tref))
	return temp.T


def get_xb(yc, zb):
	"""
    Function to compute intermediate xb vector
	:return:
	"""
	x = []
	for y, z in zip(yc, zb):
		x.append(np.cross(y.flatten(), z.flatten()))
	return np.vstack(x) / np.linalg.norm(np.vstack(x))


def get_yb(zb, xb):
	"""
    Function to compute intermediate yb vector
	:return:
	"""
	r = []
	for z, x in zip(zb, xb):
		r.append(np.cross(z.flatten(), x.flatten()))
	return np.vstack(r)


def get_zb(T):
	"""
    Function to compute intermediate zb vector
	:param T:
	:return:
	"""
	return T/np.linalg.norm(T, axis=0)


def compute_acc(cur_vel, prev_vel, time):
    """
    Assume velocity to be linear for each segment and compute acc
    :return:
    """
    return (cur_vel - prev_vel)/time


def compute_jerk(cur_acc, prev_acc, time):
    """
    Assume velocity to be linear for each segment and compute acc
    :return:
    """
    return (cur_acc - prev_acc)/time


def compute_yaw_dot(cur_yaw, prev_yaw, time):
    """
    Assume velocity to be linear for each segment and compute acc
    :return:
    """
    return (cur_yaw - prev_yaw)/time


def load_pickle(str):

    with open(str, 'rb') as handle:
        return pickle.load(handle)


def compute_traj(bag, mav_name, str, rho=1, horizon=300, full_state=False):
    """
    Function to compute the trajectories and the reward (MDP)
    :return:
    """
    pos = list()
    vel = list()
    acc = list()
    jerk = list()
    yaw = list()
    yaw_dot = list()
    # actual_yaw = list() # Need to this to compute trajectory

    time_stamps = load_pickle(str)
    print(time_stamps)
    time_idx = 0
    times = []
    count = 0
    #time_stamps = load_pickle(str)
    #print(time_stamps)
    #time_idx = 0
    #times = []
    #count = 0

    actual_traj = list()
    ref_traj = list()
    input_traj = list()


    prev_vel = np.zeros(3)
    prev_acc = np.zeros(3)
    prev_yaw = 0

    count = 0
    times = []
    print(mav_name)

    print(bag.read_messages('/' + mav_name + '/odom'))

    for topic, msg, t in bag.read_messages(topics=['/' + mav_name + '/odom', '/' + mav_name + '/position_cmd', '/' + mav_name + '/so3_cmd']):


        if topic == '/' + mav_name + '/odom':

            quarternion = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
                                msg.pose.pose.orientation.w]
            (cur_roll, cur_pitch, cur_yaw) = tf.transformations.euler_from_quaternion(quarternion)
            cur_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
            cur_acc = compute_acc(cur_vel, prev_vel, time=0.01)
            cur_jerk = compute_jerk(cur_acc, prev_acc, time=0.01)
            cur_yaw_dot = compute_yaw_dot(cur_yaw, prev_yaw, time=0.01)
            prev_acc = cur_acc
            prev_vel = cur_vel
            prev_yaw = cur_yaw
            if full_state:
                actual_traj.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
                                msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,
                                cur_acc[0], cur_acc[1], cur_acc[2], cur_jerk[0], cur_jerk[1], cur_jerk[2],
                                cur_yaw, cur_yaw_dot])
            else:
                actual_traj.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, cur_yaw])


        elif topic == '/' + mav_name + '/position_cmd':
            count += 1
            pos.append([msg.position.x, msg.position.y, msg.position.z])
            vel.append([msg.velocity.x, msg.velocity.y, msg.velocity.z])
            acc.append([msg.acceleration.x, msg.acceleration.y, msg.acceleration.z])
            jerk.append([msg.jerk.x, msg.jerk.y, msg.jerk.z])
            if msg.yaw < 0:
                yaw.append(2*np.pi + msg.yaw)
            else:
                yaw.append(msg.yaw)
            yaw_dot.append(msg.yaw_dot)

            """if time_idx < len(time_stamps):
                if t > time_stamps[time_idx]:
                    time_idx += 1
                    times.append(count-1)"""

        elif topic == '/' + mav_name + '/so3_cmd':
            # Is thrust the norm of force or the z component ?
            input_traj.append([np.linalg.norm(np.array([msg.force.x, msg.force.y, msg.force.z])), msg.angular_velocity.x,
                          msg.angular_velocity.y, msg.angular_velocity.z])


    T = get_T(acc)
    zb = get_zb(T)
    yc = get_yc(yaw)
    xb = get_xb(yc, zb)
    yb = get_yb(zb, xb)

    q = []
    for x, y, z in zip(xb, yb, zb):
        R = np.vstack([x, y, z]).T
        q.append(rotationMatrixToQuaternion1(R))

    for p, v, a, j, y, yd in zip(pos, vel, acc, jerk, yaw, yaw_dot):
        if full_state:
            ref_traj.append(np.append(np.concatenate([p, v, a, j]), [y, yd]))
        else:
            ref_traj.append(np.append(p, y))

    # Cost of the trajectory is computed as l2 norm squared between only x, y, z, yaw

    ref_traj = np.array(ref_traj)
    actual_traj = np.array(actual_traj)
    input_traj = np.array(input_traj)

    # idx = [0, 1, 2]
    cost_traj = compute_cum_tracking_cost(ref_traj, actual_traj, input_traj, horizon, horizon, rho)
    print(cost_traj.shape)
    # cost_traj = rho * np.square(np.linalg.norm(ref_traj[:, idx] - actual_traj[:, idx], axis=1)) + 0.1 * np.square(np.linalg.norm(input, axis=1)) #+ np.square(np.mod(np.abs(ref_traj[:, -2] - actual_traj[:, -2]), np.pi))\

    print(ref_traj)
    return ref_traj, actual_traj, input_traj, cost_traj, times


def reformat_traj_list(xtrajs, utrajs, rtrajs):
    """

    :return:
    """
    traj = []
    for xtraj, utraj, rtraj in zip(xtrajs, utrajs, rtrajs):
        T = len(utraj)
        traj += list(zip(xtraj[:T], utraj, rtraj, xtraj[1:]))
    return traj


if __name__ == '__main__':

    # Load bag
    bag = load_bag('./traj1.bag')

    # Compute trajectories
    ref_traj, xtrajs, utrajs, rtrajs = compute_traj(bag)

    bag.close()

    # Construct augmented states
    actual_traj = xtrajs[len(xtrajs)-len(ref_traj):]

    time_steps = 1000

    ref_chunks = [ref_traj[x:x + time_steps] for x in range(0, len(ref_traj), time_steps)]
    input_chunks = [utrajs[x:x + time_steps] for x in range(0, len(utrajs), time_steps)]
    cost_chunks = [rtrajs[x:x + time_steps] for x in range(0, len(rtrajs), time_steps)]
    actual_chunks = [xtrajs[x:x + time_steps] for x in range(0, len(xtrajs), time_steps)]

    i = 0
    for each in actual_chunks:
        aug_state = list()
        for j in range(time_steps):
            aug_state.append(np.concatenate((each[j], np.vstack(ref_chunks[i]).flatten())))
            print("ref traj dim", len(ref_chunks[i]))
        actual_chunks[i] = aug_state
        i += 1

    # Use the augmented states to do lstd



