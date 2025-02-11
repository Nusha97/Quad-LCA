"""
Training using the bags
"""

import numpy as np
import matplotlib.pyplot as plt
from model_learning import (
    TrajDataset,
    train_model,
    eval_model,
    numpy_collate,
    save_checkpoint,
    restore_checkpoint,
)
import ruamel.yaml as yaml
import torch.utils.data as data
from flax.training import train_state
import optax
import jax
from mlp_jax import MLP
# from ficnn_jax import FICNN
# import pandas as pd
import torch

# import tf
import transforms3d.euler as euler
from itertools import accumulate
from sklearn.model_selection import train_test_split

from scipy.spatial.transform import Rotation as R
import os

cwd = os.getcwd()

gamma = 1

def compute_traj(sim_data, rho=1, horizon=501, full_state=False):
    # TODO: full state

    # get the reference trajectory
    # col W-Y position
    ref_traj_x = sim_data[:, 22]
    ref_traj_y = sim_data[:, 23]
    ref_traj_z = sim_data[:, 24]
    # col AL yaw_des
    ref_traj_yaw = sim_data[:, 37]
    ref_traj = np.vstack((ref_traj_x, ref_traj_y, ref_traj_z, ref_traj_yaw)).T
    # debug: print the first 10 ref_traj
    print("ref_traj: ", ref_traj[:10, :])

    # get the actual trajectory
    # col C-E position
    actual_traj_x = sim_data[:, 2]
    actual_traj_y = sim_data[:, 3]
    actual_traj_z = sim_data[:, 4]
    # col I-L quaternion
    actual_traj_quat = sim_data[:, 8:12]

    euler_actual = R.from_quat(actual_traj_quat).as_euler("zyx", degrees=False)
    actual_yaw = euler_actual[:, 0]
    actual_traj = np.vstack((actual_traj_x, actual_traj_y, actual_traj_z, actual_yaw)).T
    # get the cmd input
    # col BN desired thrust from so3 controller
    input_traj_thrust = sim_data[:, 65]
    # print("input_traj_thrust's shape: ", input_traj_thrust.shape)

    # get the angular velocity from odometry: col M-O
    odom_ang_vel = sim_data[:, 12:15]

    motor_speed = sim_data[:, 18:22]
    input_traj = np.sqrt(np.sum(motor_speed**2, axis=1)).reshape(-1, 1)

    # debug: print the first 10 input_traj
    print("input_traj_motorspeed: ", input_traj)

    # get the cost
    cost_traj = compute_cum_tracking_cost(
        ref_traj, actual_traj, input_traj, horizon, horizon, rho
    )
    # debug: print the first 10 cost_traj
    print("cost_traj: ", cost_traj[:10, :])

    return ref_traj, actual_traj, input_traj, cost_traj, sim_data[:, 0]


def compute_cum_tracking_cost(ref_traj, actual_traj, input_traj, horizon, N, rho):
    # print type of input
    print("ref_traj's type: ", type(ref_traj))
    print("actual_traj's type: ", type(actual_traj))
    print("input_traj's type: ", type(input_traj))

    import ipdb

    # ipdb.set_trace()

    m, n = ref_traj.shape
    num_traj = int(m / horizon)
    xcost = []
    for i in range(num_traj):
        act = actual_traj[i * horizon : (i + 1) * horizon, :]
        # act = np.append(act, act[-1, :] * np.ones((N - 1, n)))
        # act = np.reshape(act, (horizon + N - 1, n))
        r0 = ref_traj[i * horizon : (i + 1) * horizon, :]
        # r0 = np.append(r0, r0[-1, :] * np.ones((N - 1, n)))
        # r0 = np.reshape(r0, (horizon + N - 1, n))
        distances = np.linalg.norm(np.diff(r0[:horizon, :3], axis=0), axis=1)
        unit_vec = np.zeros((horizon - 1, 3))

        for j in range(distances.shape[0]):
            if distances[j] > 0:
                unit_vec[j, :] = np.diff(r0[j : j + 2, :3], axis=0) / distances[j]
            else:
                unit_vec[j, :] = unit_vec[j - 1, :]
        # unit_vec = np.diff(r0[:horizon, :3], axis=0) / np.linalg.norm(
        #     np.diff(r0[:horizon, :3], axis=0), axis=1
        # ).reshape(-1, 1)
        print("unit_vec: ", unit_vec)
        des_yaw = angle_wrap(np.arctan2(unit_vec[:, 1], unit_vec[:, 0]))
        des_yaw = np.append(des_yaw, des_yaw[-1])
        print("des_yaw: ", des_yaw)

        xcost.append(
            rho
            * (
                np.linalg.norm(act[:, :3] - r0[:, :3], axis=1) ** 2
                + angle_wrap(act[:, 3] - r0[:, 3]) ** 2
                # ignore the yaw error
            )
            # input_traj cost is sum of squares of motor speeds for 4 individual motors
            + 0.001 * (1 / horizon) * np.linalg.norm(input_traj[i]) ** 2
            # + np.linalg.norm(input_traj[i]) ** 2  # Removed 0.1 multiplier
        )

    xcost.reverse()
    cost = []
    for i in range(num_traj):
        tot = list(accumulate(xcost[i], lambda x, y: x * gamma + y))
        # tot[-1] += 100 * np.linalg.norm(des_yaw - act[:horizon, 3]) ** 2
        cost.append(np.log(tot[-1]))
        # cost.append(tot[-1])
    cost.reverse()
    return np.vstack(cost)


def angle_wrap(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def main():
    rho = 1
    drag_coeff = 4

    with open(
        # r"/home/user/code/quadrotor-drag-exp/AeroWrenchPlanner/learning/params.yaml"
        r"/workspace/AeroWrenchPlanner/learning/params.yaml"
    ) as f:
        yaml_data = yaml.load(f, Loader=yaml.RoundTripLoader)

    num_hidden = yaml_data["num_hidden"]
    batch_size = yaml_data["batch_size"]
    learning_rate = yaml_data["learning_rate"]
    num_epochs = yaml_data["num_epochs"]
    model_save = yaml_data["save_path"] + str(rho) + str(drag_coeff)
    # Construct augmented states
    """
    cost_traj = cost_traj.ravel()
    # # exp_log_cost_traj into cost_traj
    # cost_traj = np.exp(cost_traj)

    # print("Costs", cost_traj)
    print("Costs shape", cost_traj.shape)

    # scatter plot for cost_traj vs index for fixed yaw
    plt.figure()
    plt.scatter(range(len(cost_traj)), np.exp(cost_traj), color="b", label="Cost")
    plt.xlabel("Trajectory Index")
    plt.ylabel("Cost")
    plt.legend()
    plt.title("Cost vs Trajectory Index for fixed yaw")
    # plt.savefig("./plots/cost" + str(rho) + ".png")
    plt.show()

    # plot cost of trajectories with different radii ranging from 1 to 5.5 and varying yaw angles from 0 to 2pi in 3d
    # # 3d scatter plot for cost_traj vs radius vs yaw for fixed yaw
    # radius = np.linspace(2, 5.5, 10)
    # yaw = np.linspace(0, 2 * np.pi, 10)
    # radius, yaw = np.meshgrid(radius, yaw)
    # cost_traj = cost_traj.reshape(10, 10)

    # # scatter plot for cost_traj vs radius for flexible yaw
    # radius = np.linspace(2, 5.5, 10)
    # plt.figure()
    # plt.scatter(radius, cost_traj, color="b", label="Cost")
    # plt.xlabel("Radius")
    # plt.ylabel("Cost")
    # plt.legend()
    # plt.title("Cost vs Radius for flexible yaw")
    # # plt.savefig("./plots/cost" + str(rho) + ".png")
    # plt.show()

    num_traj = int(len(ref_traj) / horizon)
    print("len(ref_traj): ", len(ref_traj))
    print("num_traj: ", num_traj)

    # Create augmented state

    aug_state = []
    for i in range(num_traj):
        r0 = ref_traj[i * horizon : (i + 1) * horizon, :]
        act = actual_traj[i * horizon : (i + 1) * horizon, :]
        aug_state.append(np.append(act[0, :], r0))
        # act[0, :] is the first row of actual_traj, which is the initial state, dimension is 4
        # r0 is the reference trajectory, dimension is 502*4

    aug_state = np.array(aug_state)
    print("aug_state.shape: ", aug_state.shape)  # (2, 2012)

    Tstart = 0
    Tend = aug_state.shape[0]

    p = aug_state.shape[1]
    q = 4

    print(aug_state.shape)
    """
    # Path to your CSV file
    # csv_file_path = "/workspace/data_output/data_diff_rho.csv"
    csv_file_path = cwd + "/../../data/data_diff_rho_drag" + str(drag_coeff) + ".csv"
    # csv_file_path = "/home/user/code/quadrotor-drag-exp/data/data_diff_rho.csv"

    train_dataset = TrajDataset(file_path=csv_file_path, feature_range=(-1, 1))

    # Split the dataset into training and testing subsets
    train_data, test_data = train_test_split(
        train_dataset,
        test_size=0.2,  # Specify the proportion of the dataset to use for testing (e.g., 0.2 for 20%)
        random_state=42,  # Set a random seed for reproducibility
    )

    print("Training data length: ", len(train_data))
    print("Testing data length: ", len(test_data))

    # Initialize the model
    number_of_coefficients = train_dataset.num_coefficients()
    p = number_of_coefficients  # Set this to the number of coefficients in your dataset
    print("Number of coefficients:", number_of_coefficients)

    rng = jax.random.PRNGKey(427)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (batch_size, p))  # Batch size 64, input size p
    # Initialize the model
    model = MLP(num_hidden=num_hidden, num_outputs=1)
    # model = FICNN(num_hidden_c=num_hidden, num_outputs=1, input_features_c=p, seed=rng)
    params = model.init(init_rng, inp)

    # Printing the model shows its attributes
    print(model)

    # rng = jax.random.PRNGKey(427)
    # rng, inp_rng, init_rng = jax.random.split(rng, 3)
    # inp = jax.random.normal(inp_rng, (batch_size, p))  # Batch size 64, input size p
    # Initialize the model
    # params = model.init(init_rng, inp)

    optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)
    # optimizer = optax.adam(learning_rate=learning_rate, b1=0.9, b2=0.999, eps=1e-08)

    model_state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    train_data_loader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate
    )
    trained_model_state = train_model(
        model_state, train_data_loader, num_epochs=num_epochs
    )

    # Evaluation of trained network

    eval_model(trained_model_state, train_data_loader, batch_size)

    trained_model = model.bind(trained_model_state.params)
    # save checkpoint
    save_checkpoint(trained_model_state, model_save, 7)

    # restore checkpoint
    # trained_model_state = restore_checkpoint(model_state, model_save, 7)
    # trained_model = model.bind(trained_model_state.params)

    num_batches = len(train_data_loader)
    print('Number of batches:', num_batches)
    print('Expected number of batches:', np.ceil(100000 / batch_size))  # Assuming batch_size is the size of each batch

    # Save plot on entire test dataset
    out = []
    true = []
    for batch in train_data_loader:
        data_coeffs, cost = batch  # Unpack the batch into coefficients and cost
        predicted_cost = trained_model(data_coeffs)  # Get model predictions
        # out.append(predicted_cost)
        # true.append(cost)
        # print("predicted_cost shape:", predicted_cost.shape)
        # print("cost shape:", cost.shape)
        
        out.append(predicted_cost.reshape(-1, 1))  # Reshape to ensure consistent dimension
        true.append(cost.reshape(-1, 1))  # Reshape to ensure consistent dimension
        

        # print("Cost", cost)
        # print("predicted_cost", predicted_cost)

    print("out's shape", len(out))
    out = np.vstack(out)
    print("out's shape", out.shape)
    true = np.vstack(true)
    print("true's shape", true.shape)

    ## Plotting and saving trajectories for each trial file
    plots_dir = '/workspace/data/plots_train/'
    # plots_dir = "/home/user/code/quadrotor-drag-exp/plots_train/"
    # plots_dir = cwd + "/../../data/plots_train/" 
    # plots_dir = cwd + "/../../data/plots_train/" 
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    rho_value = str(rho)  # Assuming rho is a variable you've defined earlier

    # Plotting the histogram of errors
    # Assuming 'out' is your predictions and 'true' is the actual values
    errors = np.linalg.norm(true - out, axis=1)  # Calculate the L2 norm of the errors

    # Plotting the histogram of errors
    plt.figure()
    plt.hist(errors, bins=50, alpha=0.75, color='blue')  # Adjust bins as needed
    plt.xlabel('Error (L2 norm of actual - predictions)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prediction Errors in Training Data')
    plt.grid(True)

    # Save the histogram
    # plots_dir = '/workspace/data_output/plots_train/'
    # plots_dir = "/home/user/code/quadrotor-drag-exp/plots_train/"
    rho_value = str(rho)  # Assuming rho is a variable you've defined earlier
    file_name = plots_dir + rho_value + str(drag_coeff) + 'mlp_error_hist_train.png'
    plt.savefig(file_name)
    plt.close()  # Close the plot

    # Calculate the percentage error relative to the true cost
    # Avoid division by zero by adding a small epsilon where true cost is zero
    epsilon = 1e-8
    true_flat = true.flatten()  # Flatten to ensure it is one-dimensional
    percentage_errors = (errors / (true_flat + epsilon)) * 100

    # Assuming 'percentage_errors' is the array containing your percentage error data.

    # Define the range and number of bins for the histogram
    error_range = (0, 100)  # Focus on errors between 0% and 100%
    number_of_bins = 200    # Increase the number of bins for more granularity

    # Define tick marks for the x-axis
    ticks = np.arange(error_range[0], error_range[1]+1, 5)  # Create ticks every 5%

    # Plot histogram of the actual cost values
    plt.figure()
    plt.hist(true, bins=50, alpha=0.75, color='blue')
    plt.xlabel('Cost Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Cost Values in Training Data')
    plt.grid(True)

    # Save the histogram of cost values
    cost_hist_file_name = plots_dir + rho_value + str(drag_coeff) + 'mlp_cost_values_hist_train.png'
    plt.savefig(cost_hist_file_name)
    plt.close()  # Close the plot

    # Plot histogram of the percentage errors
    plt.figure(figsize=(10, 6))
    plt.hist(percentage_errors, bins=number_of_bins, range=error_range, alpha=0.75, color='green')
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Percentage Errors Relative to True Cost in Training Data')
    plt.xticks(ticks)  # Set the ticks on the x-axis
    plt.xlim(error_range)  # Zoom in on the [0, 100] range on the x-axis
    plt.grid(True)
    plt.tight_layout()  # Adjust the layout to fit the figure nicely

    # Save the histogram of percentage errors
    error_hist_file_name = plots_dir + rho_value + str(drag_coeff) + 'mlp_percentage_error_hist_train.png'
    plt.savefig(error_hist_file_name)
    plt.close()  # Close the plot


    """
    # scatter plot
    plt.figure()
    plt.scatter(range(len(out)), out.ravel(), color="b", label="Predictions")
    plt.scatter(range(len(true)), true.ravel(), color="r", label="Actual")
    plt.xlabel("Trajectory Index")
    plt.ylabel("Cost")
    plt.legend()
    plt.title("Predicted vs Actual - Training Dataset")
    file_name = plots_dir + rho_value + '_train_actual_pred.png'

    # After your plotting code
    plt.savefig(file_name)    
    plt.show()
    # plt.figure()
    # plt.scatter(range(len(out)), out.ravel(), color="b", label="Predictions")
    # plt.scatter(range(len(true)), true.ravel(), color="r", label="Actual")
    # plt.xlabel("Trajectory Index")
    # plt.ylabel("Cost")
    # plt.legend()
    # plt.title("Predicted vs Actual - Training Dataset")
    # # Set the y-axis range from 0 to 12
    # plt.ylim(0, 12)
    # # plt.savefig("./plots/inference"+str(rho)+".png")
    # plt.show()

    # scatter plot
    # plt.figure()
    # plt.scatter(range(len(true)), true.ravel(), color="r", label="Actual")
    # plt.xlabel("Trajectory Index")
    # plt.ylabel("Cost")
    # plt.legend()
    # plt.title("Actual - Training Dataset")
    # # plt.savefig("./plots/inference"+str(rho)+".png")
    # plt.show()
    plt.figure()
    plt.scatter(range(len(true)), true.ravel(), color="r", label="Actual")
    plt.xlabel("Trajectory Index")
    plt.ylabel("Cost")
    plt.legend()
    plt.title("Actual - Training Dataset")
    # Set the y-axis range from 0 to 500
    file_name = plots_dir + rho_value + '_train_actual.png'

    # After your plotting code
    plt.savefig(file_name) 
    plt.show()

    # Two boxplots
    plt.figure()
    plt.boxplot([out.ravel(), true.ravel()], labels=["Predictions", "Actual"])
    plt.xlabel("Trajectory Index")
    plt.ylabel("Cost")
    plt.title("Predicted vs Actual - Train Dataset")
    file_name = plots_dir + rho_value + '_train_actual_pred_box.png'

    # After your plotting code
    plt.savefig(file_name) 
    plt.show()
    """
    # Evaluation of test and train dataset
    
    test_data_loader = data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate
    )

    eval_model(trained_model_state, test_data_loader, batch_size)

    # Save plot on entire test dataset
    out_test = []
    true_test = []
    for batch in test_data_loader:
        data_coeffs, cost = batch  # Unpack the batch into coefficients and cost
        predicted_cost = trained_model(data_coeffs)  # Get model predictions
        # out_test.append(predicted_cost)
        # true_test.append(cost)
        # print("predicted_cost shape:", predicted_cost.shape)
        # print("cost shape:", cost.shape)
        out_test.append(predicted_cost.reshape(-1, 1))  # Reshape to ensure consistent dimension
        true_test.append(cost.reshape(-1, 1))  # Reshape to ensure consistent dimension
                

    out_test = np.vstack(out_test)
    true_test = np.vstack(true_test)

    print(out_test.shape)
    print(true_test.shape)

    ## histogram of errors
    # Assuming 'out_test' is your predictions and 'true_test' is the actual values
    errors = np.linalg.norm(true_test - out_test, axis=1)  # Calculate the L2 norm of the errors

    # Plotting the histogram of errors
    plt.figure()
    plt.hist(errors, bins=50, alpha=0.75, color='blue')  # Adjust bins as needed
    plt.xlabel('Error (L2 norm of actual - predictions)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prediction Errors in Test Data')
    plt.grid(True)

    # Save the histogram
    # plots_dir = '/workspace/data_output/plots_train/'
    # plots_dir = "/home/user/code/quadrotor-drag-exp/plots_train/"
    rho_value = str(rho)  # Assuming rho is a variable you've defined earlier
    file_name = plots_dir + rho_value + str(drag_coeff) +'mlp_error_hist_test.png'
    plt.savefig(file_name)
    plt.close()  # Close the plot

    # # Show the plot
    # plt.show()

    # Calculate the percentage error relative to the true cost
    true_test_flat = true_test.flatten()  # Flatten to ensure it is one-dimensional
    percentage_errors = (errors / (true_test_flat + epsilon)) * 100


    # Plot histogram of the actual cost values
    plt.figure()
    plt.hist(true_test, bins=50, alpha=0.75, color='blue')
    plt.xlabel('Cost Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Cost Values in Test Data')
    plt.grid(True)

    # Save the histogram of cost values
    cost_hist_file_name = plots_dir + rho_value + str(drag_coeff) + 'mlp_cost_values_hist_test.png'
    plt.savefig(cost_hist_file_name)
    plt.close()  # Close the plot

    # Plot histogram of the percentage errors
    plt.figure(figsize=(10, 6))
    plt.hist(percentage_errors, bins=number_of_bins, range=error_range, alpha=0.75, color='green')
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Percentage Errors Relative to True Cost in Test Data')
    plt.xticks(ticks)  # Set the ticks on the x-axis
    plt.xlim(error_range)  # Zoom in on the [0, 100] range on the x-axis
    plt.grid(True)
    plt.tight_layout()  # Adjust the layout to fit the figure nicely

    # Save the histogram of percentage errors
    error_hist_file_name = plots_dir + rho_value + str(drag_coeff) + 'mlp_percentage_error_hist_test.png'
    plt.savefig(error_hist_file_name)
    plt.close()  # Close the plot

    # Convert everything to one-dimensional arrays for plotting
    out_flat = out.flatten()
    out_test_flat = out_test.flatten()

    # Scatter plot of predicted vs. true cost
    plt.figure(figsize=(10, 8))
    plt.scatter(out_flat, true_flat, alpha=0.5, color='blue', s=2, label='Training Data')
    plt.scatter(out_test_flat, true_test_flat, alpha=0.5, color='green', s=2, label='Test Data')

    # Plot y=x line indicating perfect predictions
    max_cost = max(true_flat.max(), out_flat.max(), true_test_flat.max(), out_test_flat.max())
    plt.plot([0, max_cost], [0, max_cost], 'r--', label='Perfect Prediction')

    plt.xlabel('Predicted Cost')
    plt.ylabel('True Cost')
    plt.title('Predicted vs. True Cost for Training and Test Data')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()

    # Save the scatter plot
    scatter_plot_file_name = plots_dir + rho_value + str(drag_coeff) + 'mlp_predicted_vs_true_cost.png'
    plt.savefig(scatter_plot_file_name)
    plt.close()
    """
    # scatter plot
    plt.figure()
    plt.scatter(range(len(out)), out.ravel(), color="b", label="Predictions")
    plt.scatter(range(len(true)), true.ravel(), color="r", label="Actual")
    plt.xlabel("Trajectory Index")
    plt.ylabel("Cost")
    # plt.plot(out.ravel(), "b-", label="Predictions")
    # plt.plot(true.ravel(), "r--", label="Actual")
    plt.legend()
    plt.title("Predicted vs Actual - Test Dataset")
    # plt.savefig("./plots/inference"+str(rho)+".png")
    file_name = plots_dir + rho_value + '_test_actual_pred.png'

    # After your plotting code
    plt.savefig(file_name) 
    plt.show()

    # # line plot
    # plt.figure()
    # plt.xlabel("Trajectory Index")
    # plt.ylabel("Cost")
    # plt.plot(out.ravel(), "b-", label="Predictions")
    # plt.plot(true.ravel(), "r--", label="Actual")
    # plt.legend()
    # plt.title("Predicted vs Actual - Test Dataset")
    # # plt.savefig("./plots/inference"+str(rho)+".png")
    # plt.show()

    # Two boxplots
    plt.figure()
    plt.boxplot([out.ravel(), true.ravel()], labels=["Predictions", "Actual"])
    plt.xlabel("Trajectory Index")
    plt.ylabel("Cost")
    plt.title("Predicted vs Actual - Test Dataset")
    file_name = plots_dir + rho_value + '_test_actual_pred_plot.png'

    # After your plotting code
    plt.savefig(file_name) 
    plt.show()
    """
    # eval_model(trained_model_state, test_data_loader, batch_size)
    


if __name__ == "__main__":
    main()
