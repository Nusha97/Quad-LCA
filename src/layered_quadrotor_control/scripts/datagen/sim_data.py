"""
Training using the bags
"""

from learning import lstd, valuefunc, trajdata
from env import linearenv, controller
from exputils import relerr
from trajgen import quadratic, nonlinear
from construct_traj_list import *
import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt
import torch


def main():
    # Load bag
    sim_data = load_bag('/home/anusha/2022-09-27-11-49-40.bag')
    ref_traj, actual_traj, input_traj, cost_traj = compute_traj(sim_data)
    sim_data.close()

    # Construct augmented states
    time_steps = 300
    gamma = 0.99

    idx = [0, 1, 2, 12]

    cost_traj = cost_traj.ravel()

    ref_chunks = np.array([ref_traj[x:x + time_steps] for x in range(0, len(ref_traj) - time_steps)])
    print(ref_chunks.shape)
    aug_state = []
    for xa, rref in zip(actual_traj[:-time_steps], ref_chunks):
        aug_state.append(np.concatenate([xa[idx], rref[:, idx].flatten()]))
    aug_state = np.array(aug_state)
    print(aug_state.shape)

    dataset = trajdata.TrajDataset(aug_state[:-1], input_traj[:-1],
                                   cost_traj[:-1 - time_steps, None], aug_state)
    dataset.cuda()

    p = aug_state.shape[1]
    q = 4

    mlpvalue = valuefunc.MLPValueFunc(p, [300, 300, 1])
    mlpvalue.network = mlpvalue.network.cuda()
    mlpvalue.learn(dataset, gamma, num_epoch=1000, batch_size=64, verbose=True, print_interval=50, lr=0.0005)

    torch.save(mlpvalue, '/home/anusha/mlp_model.pt')


if __name__ == '__main__':
    main()

