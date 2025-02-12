import numpy as np
from construct_traj_list import *
import pickle


def main():
    num_bags = 5
    traj = []

    for i in range(num_bags):
        bag = load_bag('/home/anusha/Research/ws_kr/src/multi_mav_manager/kr_multi_mav_manager/scripts/nn_lissajous_'+str(i)+'.bag')
        ref_traj, actual_traj, input_traj, cost_traj = compute_traj(bag, False)
        traj.append([np.array(ref_traj), np.array(actual_traj), np.array(input_traj)])

    # Save list to a pickle file
    with open('/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/traj_nn_data.pickle', 'wb') as handle:
        pickle.dump(traj, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()