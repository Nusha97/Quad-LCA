"""
Training using the bags
"""

from construct_traj_list import *
import numpy as np
import matplotlib.pyplot as plt
from model_learning import TrajDataset, train_model, eval_model, numpy_collate, save_checkpoint, restore_checkpoint
import ruamel.yaml as yaml
import torch.utils.data as data
from flax.training import train_state
import optax
import jax
from mlp_jax import MLP
from generate_data import save_object


def main():
    horizon = 300
    rho = 10
    gamma = 1
    # Load bag
    sim_data = load_bag('/home/anusha/IROS_bags/2022-09-27-11-49-40.bag')
    #sim_data = load_bag("/home/anusha/2023-02-27-13-35-15.bag")
    #sim_data = load_bag("/home/anusha/dragonfly1-2023-04-12-12-18-27.bag")

    ref_traj, actual_traj, input_traj, cost_traj, times = compute_traj(sim_data, "dragonfly1", "/home/anusha/min_jerk_times.pkl", rho)
    sim_data.close()

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    # ttraj = actual_traj.copy()
    # axes = plt.gca(projection='3d')
    axes.plot3D(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2])
    axes.set_xlim(-1, 1)
    axes.set_zlim(0, 4)
    axes.set_ylim(-1, 1)
    # axes.plot3D(waypt[0, :], waypt[1, :], waypt[2, :], '*')

    plt.show()

    with open(r"/home/anusha/Research/ws_kr/src/layered_ref_control/src/layered_ref_control/data/params.yaml") as f:
        yaml_data = yaml.load(f, Loader=yaml.RoundTripLoader)

    num_hidden = yaml_data['num_hidden']
    batch_size = yaml_data['batch_size']
    learning_rate = yaml_data['learning_rate']
    num_epochs = yaml_data['num_epochs']
    model_save = yaml_data['save_path']+str(rho)

    # Construct augmented states

    cost_traj = cost_traj.ravel()

    print("Costs", cost_traj)

    num_traj = int(len(ref_traj)/horizon)

    # Create augmented state

    aug_state = []
    for i in range(num_traj):
        r0 = ref_traj[i * horizon:(i + 1) * horizon, :]
        act = actual_traj[i * horizon:(i + 1) * horizon, :]
        aug_state.append(np.append(act[0, :], r0))

    aug_state = np.array(aug_state)
    print(aug_state.shape)

    Tstart = 0
    Tend = aug_state.shape[0]


    train_dataset = TrajDataset(aug_state[Tstart:Tend - 1, :].astype('float64'),
                          input_traj[Tstart:Tend - 1, :].astype('float64'),
                          cost_traj[Tstart:Tend - 1, None].astype('float64'),
                          aug_state[Tstart + 1:Tend, :].astype('float64'))

    p = aug_state.shape[1]
    q = 4

    print(aug_state.shape)

    model = MLP(num_hidden=num_hidden, num_outputs=1)
    # Printing the model shows its attributes
    print(model)

    rng = jax.random.PRNGKey(427)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (batch_size, p))  # Batch size 64, input size p
    # Initialize the model
    params = model.init(init_rng, inp)

    optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)
    #optimizer = optax.adam(learning_rate=learning_rate)

    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                                params=params,
                                                tx=optimizer)

    
    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)
    trained_model_state = train_model(model_state, train_data_loader, num_epochs=num_epochs)

    # Train on 2nd dataset
    sim_data = load_bag('/home/anusha/rho01.bag')
    #sim_data = load_bag("/home/anusha/dragonfly2-2023-04-12-12-18-27.bag")
    #sim_data = load_bag("/home/anusha/polynomials.bag")

    ref_traj, actual_traj, input_traj, cost_traj, times = compute_traj(sim_data, "dragonfly1",
                                                                       "/home/anusha/min_jerk_times.pkl", rho)
    sim_data.close()

    # Construct augmented states

    cost_traj = cost_traj.ravel()

    print("Costs", cost_traj)

    num_traj = int(len(ref_traj) / horizon)

    # Create augmented state

    aug_state = []
    for i in range(num_traj):
        r0 = ref_traj[i * horizon:(i + 1) * horizon, :]
        act = actual_traj[i * horizon:(i + 1) * horizon, :]
        aug_state.append(np.append(act[0, :], r0))

    aug_state = np.array(aug_state)
    print(aug_state.shape)

    Tstart = 0
    Tend = aug_state.shape[0]

    train_dataset = TrajDataset(aug_state[Tstart:Tend - 1, :].astype('float64'),
                                input_traj[Tstart:Tend - 1, :].astype('float64'),
                                cost_traj[Tstart:Tend - 1, None].astype('float64'),
                                aug_state[Tstart + 1:Tend, :].astype('float64'))

    p = aug_state.shape[1]
    q = 4

    print(aug_state.shape)

    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)
    trained_model_state = train_model(trained_model_state, train_data_loader, num_epochs=num_epochs)

    # Evaluation of network

    eval_model(trained_model_state, train_data_loader, batch_size)

    save_checkpoint(trained_model_state, model_save)

    trained_model_state = restore_checkpoint(model_state, model_save)

    trained_model = model.bind(trained_model_state.params)

    # Inference on bag2

    #inf_data = load_bag('/home/anusha/rho01.bag')
    #inf_data = load_bag("/home/anusha/IROS_bags/2023-02-27-13-35-15.bag")
    inf_data = load_bag("/home/anusha/dragonfly2-2023-04-12-12-18-27.bag")

    ref_traj, actual_traj, input_traj, cost_traj, times = compute_traj(inf_data, "dragonfly2", "/home/anusha/min_jerk_times.pkl", rho)
    inf_data.close()

    # Construct augmented states
    horizon = 300
    gamma = 1

    idx = [0, 1, 2, 12]

    cost_traj = cost_traj.ravel()

    num_traj = int(len(ref_traj) / horizon)

    # Create augmented state

    aug_state = []
    for i in range(num_traj):
        r0 = ref_traj[i * horizon:(i + 1) * horizon, :]
        act = actual_traj[i * horizon:(i + 1) * horizon, :]
        aug_state.append(np.append(act[0, :], r0))

    aug_state = np.array(aug_state)
    print(aug_state.shape)

    Tstart = 0
    Tend = aug_state.shape[0]

    test_dataset = TrajDataset(aug_state[Tstart:Tend - 1, :].astype('float64'),
                               input_traj[Tstart:Tend - 1, :].astype('float64'),
                               cost_traj[Tstart:Tend - 1, None].astype('float64'),
                               aug_state[Tstart + 1:Tend, :].astype('float64'))

    test_data_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate)
    eval_model(trained_model_state, test_data_loader, batch_size)

    # Save plot on entire test dataset
    out = []
    true = []
    for batch in test_data_loader:
        data_input, _, cost, _ = batch
        out.append(trained_model(data_input))
        true.append(cost)

    out = np.vstack(out)
    true = np.vstack(true)

    #save_object([out, true], "./main_data/preds_rho"+str(rho)+".png")

    plt.figure()
    plt.plot(out.ravel(), 'b-', label="Predictions")
    plt.plot(true.ravel(), 'r--', label="Actual")
    plt.legend()
    plt.title("Predictions of the trained network for different rho")
    #plt.savefig("./plots/inference"+str(rho)+".png")
    plt.show()




if __name__ == '__main__':
    main()

