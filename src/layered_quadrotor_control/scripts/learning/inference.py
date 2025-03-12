"""
To perform inference on an experiment that collected 20 trajectories and compile all data
"""

from construct_traj_list import *
import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt
import torch

from learning import lstd, valuefunc, trajdata
from env import linearenv, controller
from exputils import relerr
from trajgen import quadratic, nonlinear


def load_infbags(str1, str2):
    # Load inference bags with Lissajous curves
    bag = load_bag(str1)
    ref_traj, actual_traj, input_traj, cost_traj, times = compute_traj(bag, str2)
    bag.close()
    return ref_traj, actual_traj, input_traj, cost_traj, times


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


def main():
    # Write a loop to iterate through different bags and compute costs and make plots
    rho = [0.1, 10]
    traj_costs = []
    for i in range(len(rho)):
        if i == 0:
            ref_traj, actual_traj, input_traj, cost_traj, times = load_infbags('/home/anusha/rho01.bag', '/home/anusha/rho01.pkl')
        else:
            ref_traj, actual_traj, input_traj, cost_traj, times = load_infbags('/home/anusha/rho10.bag', '/home/anusha/rho10.pkl')
        cost = compute_tracking_cost(ref_traj, actual_traj, input_traj)
        start = times[0::2]
        end = times[1::2]
        print(start)
        print(end)
        total_cost = compute_total_cost(start, end, cost)
        traj_costs.append(total_cost)
    print(traj_costs)
    save_object(traj_costs, '/home/anusha/Downloads/icra_results/cost_rho_mlp_post_icra.pkl')

    # Make box plots for each rho value as the final plot

    # For now just plot each rho using subplot
    num_traj = 10
    fig, axes = plt.subplots(2, 1)
    for i in range(2):
        axes[i].plot(range(0, num_traj), traj_costs[i], 'r*', label='tracking cost')
        axes[i].set_title('Plot of cost for rho {:5.2f}, mean: {:6.2f}'.format(rho[i], np.mean(traj_costs[i])))
        axes[i].set_xlabel('Trajectory index')
        axes[i].set_ylabel('Total cost')
        axes[i].legend()
    """k = 0
    for i in range(2):
        for j in range():
            axes[i, j].plot(range(0, num_traj), traj_costs[k], 'r*', label='tracking cost')
            axes[i, j].set_title('Plot of cost for rho {:5.2f}, mean: {:6.2f}'.format(rho[k], np.mean(traj_costs[k])))
            axes[i, j].set_xlabel('Trajectory index')
            axes[i, j].set_ylabel('Total cost')
            axes[i, j].legend()
            k += 1"""
    fig.savefig('/home/anusha/Downloads/plots2_new.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()


