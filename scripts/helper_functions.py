"""
SYNOPSIS
    Helper functions for data generation and value function network training
DESCRIPTION

    Contains helper functions such as computing the input for the ILQR system,
    tracking costs and offseting angles to be between 0 and 2pi.
AUTHOR

    Anusha Srikanthan <sanusha@seas.upenn.edu>
LICENSE

VERSION
    0.0
"""

from itertools import accumulate
import numpy as np

gamma = 1

def compute_input(x, r, rdot, Kp, Kd=None):
    """
    Function to compute the PD control law
    """
    theta = x[2]
    phi = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    # v, w = np.matmul(np.linalg.inv(np.eye(2) - np.matmul(Kd, phi)), np.matmul(Kp, x - r) - np.matmul(Kd, rdot))
    Kd = np.linalg.pinv(phi)
    v, w = np.matmul(Kd, rdot) + np.matmul(Kp, x-r)
    return np.array([v, w])


def compute_rdot(ref, dt):
    """
    Return the numerical differentiation of ref
    :param ref:
    :param dt:
    :return:
    """
    cur_ref = ref[1:]
    prev_ref = ref[:-1]
    rdot = np.zeros(ref.shape)
    rdot[1:, :] = (cur_ref - prev_ref) / dt

    return rdot


def angle_wrap(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def compute_tracking_cost(ref_traj, actual_traj, rdot_traj, Kp, N, horizon, rho=0):
    num_traj = int(ref_traj.shape[0]/horizon)
    input_traj = []
    for i in range(len(ref_traj)):
        input_traj.append(compute_input(actual_traj[i, :], ref_traj[i, :], rdot_traj[i, :], Kp))
    # print("Inputs", input_traj)
    # input_traj = [compute_input(x, r, rdot, Kp, Kd) for x, r, rdot in
    #              zip(actual_traj, ref_traj, rdot_traj)]
    xcost = []
    for i in range(num_traj):
        act = actual_traj[i*horizon:(i+1)*horizon, :]
        act = np.append(act, act[-1, :] * np.ones((N-1, 3)))
        act = np.reshape(act, (horizon+N-1, 3))
        r0 = ref_traj[i*horizon:(i+1)*horizon, :]
        r0 = np.append(r0, r0[-1, :] * np.ones((N-1, 3)))
        r0 = np.reshape(r0, (horizon + N - 1, 3))

        xcost.append(rho * (np.linalg.norm(act[:, :2] - r0[:, :2], axis=1) ** 2 +
             angle_wrap(act[:, 2] - r0[:, 2]) ** 2) + 0.1 * np.linalg.norm(input_traj[i]) ** 2)
        # print("Ref", r0)

        #for j in range(horizon):
            # print("Cost", np.linalg.norm(actual_traj[j:j + N, :2] - ref_traj[j:j + N, :2], axis=1) ** 2 +
             # angle_wrap(actual_traj[j:j + N, 2] - ref_traj[j:j + N, 2]) ** 2)
        #    xcost.append(rho * (np.linalg.norm(act[j:j + N, :2] - r0[j:j + N, :2], axis=1) ** 2 +
        #     angle_wrap(act[j:j + N, 2] - r0[j:j + N, 2]) ** 2) + np.linalg.norm(input_traj[i]) ** 2)
    print(len(xcost))
    # xcost = [np.linalg.norm(actual_traj[i:i + N, :2] - ref_traj[i:i + N, :2], axis=1) ** 2 +
    #         angle_wrap(actual_traj[i:i + N, 2] - ref_traj[i:i + N, 2]) ** 2 for i in range(len(ref_traj) - N)]

    xcost.reverse()
    cost = []
    #for i in range(len(ref_traj) - N):
    for i in range(num_traj):
        tot = list(accumulate(xcost[i], lambda x, y: x * gamma + y))
        cost.append(np.log(tot[-1]))
    cost.reverse()
    return np.vstack(cost), np.vstack(input_traj)


def compute_cum_tracking_cost(ref_traj, actual_traj, input_traj, horizon, N, rho):
    m, n = ref_traj.shape
    num_traj = int(m / horizon)
    xcost = []
    for i in range(num_traj):
        act = actual_traj[i * horizon:(i + 1) * horizon, :]
        act = np.append(act, act[-1, :] * np.ones((N - 1, n)))
        act = np.reshape(act, (horizon + N - 1, n))
        r0 = ref_traj[i * horizon:(i + 1) * horizon, :]
        r0 = np.append(r0, r0[-1, :] * np.ones((N - 1, n)))
        r0 = np.reshape(r0, (horizon + N - 1, n))
        xcost.append(rho * (np.linalg.norm(act[:, :3] - r0[:, :3], axis=1) ** 2 +
                        angle_wrap(act[:, 3] - r0[:, 3]) ** 2) + 0.1 * np.linalg.norm(input_traj[i]) ** 2)
    #xcost = [rho * (np.linalg.norm(actual_traj[i:i + N, :2] - ref_traj[i:i + N, :2], axis=1) ** 2 +
    #         angle_wrap(actual_traj[i:i + N, 2] - ref_traj[i:i + N, 2]) ** 2 for i in range(len(ref_traj) - N)]

    xcost.reverse()
    cost = []
    for i in range(num_traj):
        tot = list(accumulate(xcost[i], lambda x, y: x * gamma + y))
        cost.append(np.log(tot[-1]))
    cost.reverse()
    return np.vstack(cost)






