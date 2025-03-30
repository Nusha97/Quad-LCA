"""
run inference on the trained model once
save the coefficients of the trajectory
"""

import csv
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
import numpy as np
import sys
import ruamel.yaml as yaml
import flax
import optax
from flax.training import train_state
import jax
from scripts.mlp_jax import MLP
from scripts.model_learning import restore_checkpoint, train_model, numpy_collate, TrajDataset
import torch.utils.data as data
from sklearn.model_selection import train_test_split 
from scipy.spatial.transform import Rotation as R
import time
from rotorpy.utils.occupancy_map import OccupancyMap
from rotorpy.controllers.quadrotor_control import SE3Control
# from rotorpy.controllers.quadrotor_l1control import L1SE3Control
from rotorpy.vehicles.multirotor import Multirotor
# from layered_quadrotor_control.scripts.inference.regularized_trajectory import RegularizedTrajectory
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.environments import Environment
from rotorpy.world import World

gamma = 1

# drag_coeff = 1

PI = np.pi


def sample_waypoints(
    num_waypoints,
    world,
    world_buffer=2,
    check_collision=True,
    min_distance=1,
    max_distance=3,
    max_attempts=1000,
    start_waypoint=None,
    end_waypoint=None,
    rng=None,
    seed=None,
):
    """
    Samples random waypoints (x,y,z) in the world. Ensures waypoints do not collide with objects, although there is no guarantee that
    the path you generate with these waypoints will be collision free.
    Inputs:
        num_waypoints: Number of waypoints to sample.
        world: Instance of World class containing the map extents and any obstacles.
        world_buffer: Buffer around the world used for sampling. This is used to ensure that waypoints are at least this distance away
            from the edge of the world.
        check_collision: If True, checks for collisions with obstacles. If False, does not check for collisions. Checking collisions slows down the script.
        min_distance: Minimum distance between waypoints consecutive waypoints.
        max_distance: Maximum distance between consecutive waypoints.
        max_attempts: Maximum number of attempts to sample a waypoint.
        start_waypoint: If specified, the first waypoint will be this point.
        end_waypoint: If specified, the last waypoint will be this point.
    Outputs:
        waypoints: A list of (x,y,z) waypoints. [[waypoint_1], [waypoint_2], ... , [waypoint_n]]
    """

    if min_distance > max_distance:
        raise Exception("min_distance must be less than or equal to max_distance.")

    def check_distance(waypoint, waypoints, min_distance, max_distance):
        """
        Checks if the waypoint is at least min_distance away from all other waypoints.
        Inputs:
            waypoint: The waypoint to check.
            waypoints: A list of waypoints.
            min_distance: The minimum distance the waypoint must be from all other waypoints.
            max_distance: The maximum distance the waypoint can be from all other waypoints.
        Outputs:
            collision: True if the waypoint is at least min_distance away from all other waypoints. False otherwise.
        """
        collision = False
        for w in waypoints:
            if (np.linalg.norm(waypoint - w) < min_distance) or (
                np.linalg.norm(waypoint - w) > max_distance
            ):
                collision = True
        return collision

    def check_obstacles(waypoint, occupancy_map):
        """
        Checks if the waypoint is colliding with any obstacles in the world.
        Inputs:
            waypoint: The waypoint to check.
            occupancy_map: An instance of the occupancy map.
        Outputs:
            collision: True if the waypoint is colliding with any obstacles in the world. False otherwise.
        """
        collision = False
        if occupancy_map.is_occupied_metric(waypoint):
            collision = True
        return collision

    def single_sample(
        world,
        current_waypoints,
        world_buffer,
        occupancy_map,
        min_distance,
        max_distance,
        max_attempts=1000,
        rng=None,
        seed=None,
    ):
        """
        Samples a single waypoint.
        Inputs:
            world: Instance of World class containing the map extents and any obstacles.
            world_buffer: Buffer around the world used for sampling. This is used to ensure that waypoints are at least this distance away
                from the edge of the world.
            occupancy_map: An instance of the occupancy map.
            min_distance: Minimum distance between waypoints consecutive waypoints.
            max_distance: Maximum distance between consecutive waypoints.
            max_attempts: Maximum number of attempts to sample a waypoint.
            rng: Random number generator. If None, uses numpy's random number generator.
            seed: Seed for the random number generator.
        Outputs:
            waypoint: A single (x,y,z) waypoint.
        """

        if seed is not None:
            np.random.seed(seed)

        num_attempts = 0

        world_lower_limits = (
            np.array(world.world["bounds"]["extents"][0::2]) + world_buffer
        )
        world_upper_limits = (
            np.array(world.world["bounds"]["extents"][1::2]) - world_buffer
        )

        if len(current_waypoints) == 0:
            max_distance_lower_limits = world_lower_limits
            max_distance_upper_limits = world_upper_limits
        else:
            max_distance_lower_limits = current_waypoints[-1] - max_distance
            max_distance_upper_limits = current_waypoints[-1] + max_distance

        lower_limits = np.max(
            np.vstack((world_lower_limits, max_distance_lower_limits)), axis=0
        )
        upper_limits = np.min(
            np.vstack((world_upper_limits, max_distance_upper_limits)), axis=0
        )

        waypoint = np.random.uniform(low=lower_limits, high=upper_limits, size=(3,))
        while check_obstacles(waypoint, occupancy_map) or (
            check_distance(waypoint, current_waypoints, min_distance, max_distance)
            if occupancy_map is not None
            else False
        ):
            waypoint = np.random.uniform(low=lower_limits, high=upper_limits, size=(3,))
            num_attempts += 1
            if num_attempts > max_attempts:
                raise Exception(
                    "Could not sample a waypoint after {} attempts. Issue with obstacles: {}, Issue with min/max distance: {}".format(
                        max_attempts,
                        check_obstacles(waypoint, occupancy_map),
                        check_distance(
                            waypoint, current_waypoints, min_distance, max_distance
                        ),
                    )
                )
        return waypoint

    ######################################################################################################################

    waypoints = []

    if check_collision:
        # Create occupancy map from the world. This can potentially be slow, so only do it if the user wants to check for collisions.
        occupancy_map = OccupancyMap(
            world=world, resolution=[0.5, 0.5, 0.5], margin=0.1
        )
    else:
        occupancy_map = None

    if start_waypoint is not None:
        waypoints = [start_waypoint]
    else:
        # Randomly sample a start waypoint.
        waypoints.append(
            single_sample(
                world,
                waypoints,
                world_buffer,
                occupancy_map,
                min_distance,
                max_distance,
                max_attempts,
                rng,
                seed,
            )
        )

    num_waypoints -= 1

    if end_waypoint is not None:
        num_waypoints -= 1

    for _ in range(num_waypoints):
        waypoints.append(
            single_sample(
                world,
                waypoints,
                world_buffer,
                occupancy_map,
                min_distance,
                max_distance,
                max_attempts,
                rng,
                seed,
            )
        )

    if end_waypoint is not None:
        waypoints.append(end_waypoint)

    return np.array(waypoints)


def sample_yaw(seed, waypoints, yaw_min=-np.pi, yaw_max=np.pi):
    """
    Samples random yaw angles for the waypoints.
    """
    np.random.seed(seed)
    yaw_angles = np.random.uniform(low=yaw_min, high=yaw_max, size=len(waypoints))

    return yaw_angles


def compute_cost_mean(sim_result):
    """
    Computes the cost from the output of a simulator instance.
    Inputs:
        sim_result: The output of a simulator instance.
    Outputs:
        cost: The cost of the trajectory.
    """

    # Some useful values from the trajectory.

    x = sim_result["state"]["x"]  # Position
    v = sim_result["state"]["v"]  # Velocity

    x_des = sim_result["flat"]["x"]  # Desired position
    v_des = sim_result["flat"]["x_dot"]  # Desired velocity

    # Cost components
    position_error = np.linalg.norm(x - x_des, axis=1).mean()
    velocity_error = np.linalg.norm(v - v_des, axis=1).mean()

    # Input cost from thrust and body moment
    # Compute total cost as a weighted sum of tracking errors
    rho_position, rho_velocity, rho_attitude, rho_thrust, rho_moment = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )  # Adjust these weights as needed
    sim_cost = rho_position * position_error + rho_velocity * velocity_error

    return sim_cost


def compute_yaw_from_quaternion(quaternions):
    R_matrices = R.from_quat(quaternions).as_matrix()
    b3 = R_matrices[:, :, 2]
    H = np.zeros((len(quaternions), 3, 3))
    for i in range(len(quaternions)):
        H[i, :, :] = np.array(
            [
                [
                    1 - (b3[i, 0] ** 2) / (1 + b3[i, 2]),
                    -(b3[i, 0] * b3[i, 1]) / (1 + b3[i, 2]),
                    b3[i, 0],
                ],
                [
                    -(b3[i, 0] * b3[i, 1]) / (1 + b3[i, 2]),
                    1 - (b3[i, 1] ** 2) / (1 + b3[i, 2]),
                    b3[i, 1],
                ],
                [-b3[i, 0], -b3[i, 1], b3[i, 2]],
            ]
        )
    Hyaw = np.transpose(H, axes=(0, 2, 1)) @ R_matrices
    actual_yaw = np.arctan2(Hyaw[:, 1, 0], Hyaw[:, 0, 0])
    return actual_yaw


def compute_cost(sim_result, robust_c=1.0):
    """
    Computes the cost from the output of a simulator instance.
    Inputs:
        sim_result: The output of a simulator instance.
    Outputs:
        cost: The cost of the trajectory.
    """

    # Some useful values from the trajectory.
    actual_pos = sim_result["state"]["x"]  # Position
    # actual_vel = sim_result['state']['v']                                    # Velocity
    # actual_q = sim_result['state']['q']                                      # Attitude
    # actual_yaw = compute_yaw_from_quaternion(actual_q)                       # Yaw angle

    des_pos = sim_result["flat"]["x"]  # Desired position
    # des_vel = sim_result['flat']['x_dot']                             # Desired velocity
    # q_des = sim_result['control']['cmd_q']                          # Desired attitude
    # desired_yaw = compute_yaw_from_quaternion(q_des)

    # cmd_thrust = sim_result['control']['cmd_thrust']                # Desired thrust
    # cmd_moment = sim_result['control']['cmd_moment']                # Desired body moment

    # Cost components by cumulative sum of squared norms
    # position_error = np.linalg.norm(actual_pos - des_pos, axis=1) ** 2

    position_error = np.linalg.norm(actual_pos - des_pos, axis=1)
    # velocity_error = np.linalg.norm(actual_vel - des_vel, axis=1)**2
    # yaw_error = (actual_yaw - desired_yaw)**2
    # print(f"yaw_error: {yaw_error}")
    # tracking_cost = position_error + velocity_error + yaw_error

    # control effort
    # thrust_error = cmd_thrust**2
    # moment_error = np.linalg.norm(cmd_moment, axis=1)**2
    # control_cost = thrust_error + moment_error

    # sim_cost = np.sum(tracking_cost + robust_c * control_cost)

    # return sim_cost
    # only position error
    # return np.sum(position_error)
    return np.max(position_error)


# Function to run simulation and compute cost
def run_simulation_and_compute_cost(
    waypoints,
    yaw_angles,
    vavg,
    # use_neural_network,
    regularizer=None,
    vehicle=None,
    controller=None,
    robust_c=1.0,
):
    traj = RegularizedTrajectory(
        points=waypoints,
        yaw_angles=yaw_angles,
        v_avg=vavg,
        # use_neural_network=use_neural_network,
        regularizer=regularizer,
    )
    nan_encountered = traj.nan_encountered  # Flag indicating if NaN was encountered

    sim_instance = Environment(
        vehicle=vehicle,
        controller=controller,
        trajectory=traj,
        wind_profile=None,
        sim_rate=100,
    )

    # Set initial state
    x0 = {
        "x": waypoints[0],
        "v": np.zeros(
            3,
        ),
        "q": np.array([0, 0, 0, 1]),  # quaternion
        "w": np.zeros(
            3,
        ),
        "wind": np.array([0, 0, 0]),
        "rotor_speeds": np.array([1788.53, 1788.53, 1788.53, 1788.53]),
    }
    sim_instance.vehicle.initial_state = x0

    waypoint_times = traj.t_keyframes
    # sim_result = sim_instance.run(t_final=traj.t_keyframes[-1], use_mocap=False, terminate=False, plot=False)
    sim_result = sim_instance.run(
        t_final=traj.t_keyframes[-1],
        use_mocap=False,
        terminate=False,
        plot=True,
        animate_bool=True,  # Boolean: determines if the animation of vehicle state will play.
        # animate_bool=False,
        animate_wind=False,  # Boolean: determines if the animation will include a wind vector.
        verbose=True,  # Boolean: will print statistics regarding the simulation.
        waypoints=waypoints,
        fname="trial_29",  # Filename is specified if you want to save the animation. Default location is the home directory.
    )
    trajectory_cost = compute_cost(sim_result, robust_c=robust_c)

    # Now extract the polynomial coefficients for the trajectory.
    pos_poly = traj.c_opt_xyz
    yaw_poly = traj.c_opt_yaw

    summary_output = np.concatenate(
        (
            np.array([trajectory_cost]),
            pos_poly.ravel(),
            yaw_poly.ravel(),
            waypoints.ravel(),
        )
    )
    return sim_result, trajectory_cost, waypoint_times, nan_encountered, summary_output


def write_to_csv(output_file, row):
    with open(output_file, "a", newline="") as file:
        writer = csv.writer(file)
        num_waypoints = 4
        # writer header
        writer.writerow(
            ["cost"]
            + [
                "x_poly_seg_{}_coeff_{}".format(i, j)
                for i in range(num_waypoints - 1)
                for j in range(8)
            ]
            + [
                "y_poly_seg_{}_coeff_{}".format(i, j)
                for i in range(num_waypoints - 1)
                for j in range(8)
            ]
            + [
                "z_poly_seg_{}_coeff_{}".format(i, j)
                for i in range(num_waypoints - 1)
                for j in range(8)
            ]
            + [
                "yaw_poly_seg_{}_coeff_{}".format(i, j)
                for i in range(num_waypoints - 1)
                for j in range(8)
            ]
            # + ['waypoints_x_seg_{}'.format(i) for i in range(num_waypoints)]
            # + ['waypoints_y_seg_{}'.format(i) for i in range(num_waypoints)]
            # + ['waypoints_z_seg_{}'.format(i) for i in range(num_waypoints)])
            # i need the col name like this: waypoints_x_seg_0, waypoints_y_seg_0, waypoints_z_seg_0, waypoints_x_seg_1, waypoints_y_seg_1, waypoints_z_seg_1...
            # first repete by x, y, z for the same segment, then repeat for the next segment
            + [
                "waypoints_{0}_seg_{1}".format(j, i)
                for i in range(num_waypoints - 1)
                for j in ["x", "y", "z"]
            ]
        )

        writer.writerow(row)
    return None


def plot_cumulative_tracking_error(sim_result_minsnap, sim_result_drag, sim_result_dragcomp, sim_result_l1, filename=None):
    # Cumulative Tracking Error for Minsnap
    error_minsnap = np.cumsum(
        np.linalg.norm(
            sim_result_minsnap["state"]["x"] - sim_result_minsnap["flat"]["x"], axis=1
        )
        ** 2
    )

    # Cumulative Tracking Error for Drag-aware
    error_drag = np.cumsum(
        np.linalg.norm(
            sim_result_drag["state"]["x"] - sim_result_drag["flat"]["x"], axis=1
        )
        ** 2
    )

    # Cumulative Tracking Error for Drag with Compensation
    error_dragcomp = np.cumsum(
        np.linalg.norm(
            sim_result_dragcomp["state"]["x"] - sim_result_dragcomp["flat"]["x"], axis=1
        )
        ** 2
    )

    # Cumulative Tracking Error for L1
    error_l1 = np.cumsum(
        np.linalg.norm(
            sim_result_l1["state"]["x"] - sim_result_l1["flat"]["x"], axis=1
        )
        ** 2
    )

    plt.figure()
    plt.plot(
        sim_result_minsnap["time"],
        error_minsnap,
        label="Minsnap",
        linewidth=4,
        color="red",
    )
    plt.plot(
        sim_result_drag["time"],
        error_drag,
        label="Drag-aware",
        linewidth=4,
        color="blue",
    )
    plt.plot(
        sim_result_dragcomp["time"],
        error_dragcomp,
        label="Minsnap with Drag Compensation",
        linewidth=4,
        color="green",
    )
    plt.plot(
        sim_result_l1["time"],
        error_l1,
        label="Minsnap with Drag Compensation",
        linewidth=4,
        color="green",
    )
    plt.xlabel("Time(s)", fontsize=18)
    plt.ylabel("Cumulative Error", fontsize=18)
    plt.title("Cumulative Tracking Error vs Time", fontsize=20)
    # put the legend on the top right
    plt.legend(fontsize=16, loc="upper left")
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")

    plt.close()
    # print the ratio of total error for sim_result_drag/minsnap
    print("final error_minsnap: ", error_minsnap[-1])
    print("final error_drag: ", error_drag[-1])
    print("Cumulative Tracking Error Ratio: ", error_drag[-1] / error_minsnap[-1])


def plot_results_with_drag(
    sim_result_minsnap,
    sim_result_nn,
    sim_result_drag,
    waypoints,
    initial_cost,
    predicted_cost,
    drag_comp_cost,
    filename=None,
    waypoints_time=None,
):
    actual_yaw_init = compute_yaw_from_quaternion(sim_result_minsnap["state"]["q"])
    actual_yaw_nn = compute_yaw_from_quaternion(sim_result_nn["state"]["q"])
    actual_yaw_drag = compute_yaw_from_quaternion(sim_result_drag["state"]["q"])

    # Create the figure
    fig = plt.figure(figsize=(18, 8))

    # 3D Trajectory plot with waypoints
    ax_traj = fig.add_subplot(121, projection="3d")
    # Minsnap trajectories
    ax_traj.plot3D(
        sim_result_minsnap["flat"]["x"][:, 0],
        sim_result_minsnap["flat"]["x"][:, 1],
        sim_result_minsnap["flat"]["x"][:, 2],
        color="lightcoral",
        linestyle="--",
        linewidth=4,
        label="Minsnap Reference",
    )
    ax_traj.plot3D(
        sim_result_minsnap["state"]["x"][:, 0],
        sim_result_minsnap["state"]["x"][:, 1],
        sim_result_minsnap["state"]["x"][:, 2],
        color="red",
        linestyle="-",
        linewidth=4,
        label="Minsnap Actual",
    )
    # Our method trajectories
    ax_traj.plot3D(
        sim_result_nn["flat"]["x"][:, 0],
        sim_result_nn["flat"]["x"][:, 1],
        sim_result_nn["flat"]["x"][:, 2],
        color="lightskyblue",
        linestyle="--",
        linewidth=4,
        label="Drag-aware Reference",
    )
    ax_traj.plot3D(
        sim_result_nn["state"]["x"][:, 0],
        sim_result_nn["state"]["x"][:, 1],
        sim_result_nn["state"]["x"][:, 2],
        color="blue",
        linestyle="-",
        linewidth=4,
        label="Drag-aware Actual",
    )
    # Drag-aware trajectories
    ax_traj.plot3D(
        sim_result_drag["flat"]["x"][:, 0],
        sim_result_drag["flat"]["x"][:, 1],
        sim_result_drag["flat"]["x"][:, 2],
        color="lightgreen",
        linestyle="--",
        linewidth=4,
        label="minsnap+drag Reference",
    )
    ax_traj.plot3D(
        sim_result_drag["state"]["x"][:, 0],
        sim_result_drag["state"]["x"][:, 1],
        sim_result_drag["state"]["x"][:, 2],
        color="green",
        linestyle="-",
        linewidth=4,
        label="minsnap+drag Actual",
    )
    # # set x, y, z limits
    # ax_traj.set_xlim(-3, 3)
    # ax_traj.set_ylim(-3, 3)
    # ax_traj.set_zlim(-3, 3)
    # Waypoints
    ax_traj.scatter(
        waypoints[:, 0],
        waypoints[:, 1],
        waypoints[:, 2],
        c="k",
        marker="o",
        s=60,
        label="Waypoints",
    )

    ax_traj.set_title("3D Trajectories with Waypoints", fontsize=20)
    ax_traj.set_xlabel("X(m)", fontsize=18)
    ax_traj.set_ylabel("Y(m)", fontsize=18)
    ax_traj.set_zlabel("Z(m)", fontsize=18)
    ax_traj.legend(fontsize=16)

    # # Set the view angle
    # ax_traj.view_init(elev=30, azim=210)   # Set the elevation and azimuth angles

    # print Minsnap Avg Cost from Sim; Drag-aware Avg Cost from Sim; minsnap+drag Avg Cost from Sim
    print("Minsnap Cost from Sim: ", initial_cost)
    print("Drag-aware Cost from Sim: ", predicted_cost)
    print("minsnap+drag Cost from Sim: ", drag_comp_cost)

    # Subplots for X, Y, Z, Yaw
    gs = fig.add_gridspec(4, 2)
    ax_x = fig.add_subplot(gs[0, 1])
    ax_y = fig.add_subplot(gs[1, 1])
    ax_z = fig.add_subplot(gs[2, 1])
    ax_yaw = fig.add_subplot(gs[3, 1])

    # Subplot for X
    ax_x.plot(
        sim_result_minsnap["time"],
        sim_result_minsnap["flat"]["x"][:, 0],
        color="lightcoral",
        linestyle="--",
        linewidth=4,
    )
    ax_x.plot(
        sim_result_minsnap["time"],
        sim_result_minsnap["state"]["x"][:, 0],
        color="red",
        linestyle="-",
        linewidth=4,
    )
    ax_x.plot(
        sim_result_nn["time"],
        sim_result_nn["flat"]["x"][:, 0],
        color="lightskyblue",
        linestyle="--",
        linewidth=4,
    )
    ax_x.plot(
        sim_result_nn["time"],
        sim_result_nn["state"]["x"][:, 0],
        color="blue",
        linestyle="-",
        linewidth=4,
    )
    ax_x.plot(
        sim_result_drag["time"],
        sim_result_drag["flat"]["x"][:, 0],
        color="lightgreen",
        linestyle="--",
        linewidth=4,
    )
    ax_x.plot(
        sim_result_drag["time"],
        sim_result_drag["state"]["x"][:, 0],
        color="green",
        linestyle="-",
        linewidth=4,
    )
    ax_x.set_title("X Position Over Time", fontsize=20)
    ax_x.set_xlabel("Time(s)", fontsize=18)
    ax_x.set_ylabel("X Position(m)", fontsize=18)
    # ax_x.legend(['Initial Actual', 'Initial Ref', 'NN Actual', 'NN Ref'], fontsize=12)

    # Subplot for Y
    ax_y.plot(
        sim_result_minsnap["time"],
        sim_result_minsnap["flat"]["x"][:, 1],
        color="lightcoral",
        linestyle="--",
        linewidth=4,
    )
    ax_y.plot(
        sim_result_minsnap["time"],
        sim_result_minsnap["state"]["x"][:, 1],
        color="red",
        linestyle="-",
        linewidth=4,
    )
    ax_y.plot(
        sim_result_nn["time"],
        sim_result_nn["flat"]["x"][:, 1],
        color="lightskyblue",
        linestyle="--",
        linewidth=4,
    )
    ax_y.plot(
        sim_result_nn["time"],
        sim_result_nn["state"]["x"][:, 1],
        color="blue",
        linestyle="-",
        linewidth=4,
    )
    ax_y.plot(
        sim_result_drag["time"],
        sim_result_drag["flat"]["x"][:, 1],
        color="lightgreen",
        linestyle="--",
        linewidth=4,
    )
    ax_y.plot(
        sim_result_drag["time"],
        sim_result_drag["state"]["x"][:, 1],
        color="green",
        linestyle="-",
        linewidth=4,
    )
    ax_y.set_title("Y Position Over Time", fontsize=20)
    ax_y.set_xlabel("Time(s)", fontsize=18)
    ax_y.set_ylabel("Y Position(m)", fontsize=18)
    # ax_y.legend(['Initial Actual', 'Initial Ref', 'NN Actual', 'NN Ref'], fontsize=12)

    # Subplot for Z
    ax_z.plot(
        sim_result_minsnap["time"],
        sim_result_minsnap["flat"]["x"][:, 2],
        color="lightcoral",
        linestyle="--",
        linewidth=4,
    )
    ax_z.plot(
        sim_result_minsnap["time"],
        sim_result_minsnap["state"]["x"][:, 2],
        color="red",
        linestyle="-",
        linewidth=4,
    )
    ax_z.plot(
        sim_result_nn["time"],
        sim_result_nn["flat"]["x"][:, 2],
        color="lightskyblue",
        linestyle="--",
        linewidth=4,
    )
    ax_z.plot(
        sim_result_nn["time"],
        sim_result_nn["state"]["x"][:, 2],
        color="blue",
        linestyle="-",
        linewidth=4,
    )
    ax_z.plot(
        sim_result_drag["time"],
        sim_result_drag["flat"]["x"][:, 2],
        color="lightgreen",
        linestyle="--",
        linewidth=4,
    )
    ax_z.plot(
        sim_result_drag["time"],
        sim_result_drag["state"]["x"][:, 2],
        color="green",
        linestyle="-",
        linewidth=4,
    )
    ax_z.set_title("Z Position Over Time", fontsize=20)
    ax_z.set_xlabel("Time(s)", fontsize=18)
    ax_z.set_ylabel("Z Position(m)", fontsize=18)
    # ax_z.legend(['Initial Actual', 'Initial Ref', 'NN Actual', 'NN Ref'], fontsize=12)

    # Adding keyframes to the subplots
    for ax, dim in zip([ax_x, ax_y, ax_z], [0, 1, 2]):
        ax.scatter(
            waypoints_time,
            waypoints[:, dim],
            c="k",
            marker="o",
            s=60,
            label="Waypoints",
        )

    # Subplot for Yaw
    ax_yaw.plot(
        sim_result_minsnap["time"],
        actual_yaw_init,
        color="red",
        linestyle="-",
        linewidth=4,
    )
    ax_yaw.plot(
        sim_result_minsnap["time"],
        sim_result_minsnap["flat"]["yaw"],
        color="lightcoral",
        linestyle="--",
        linewidth=4,
    )
    ax_yaw.plot(
        sim_result_nn["time"], actual_yaw_nn, color="blue", linestyle="-", linewidth=4
    )
    ax_yaw.plot(
        sim_result_nn["time"],
        sim_result_nn["flat"]["yaw"],
        color="lightskyblue",
        linestyle="--",
        linewidth=4,
    )
    ax_yaw.plot(
        sim_result_drag["time"],
        actual_yaw_drag,
        color="green",
        linestyle="-",
        linewidth=4,
    )
    ax_yaw.plot(
        sim_result_drag["time"],
        sim_result_drag["flat"]["yaw"],
        color="lightgreen",
        linestyle="--",
        linewidth=4,
    )

    ax_yaw.set_title("Yaw Angle Over Time", fontsize=20)
    ax_yaw.set_xlabel("Time(s)", fontsize=18)
    ax_yaw.set_ylabel("Yaw Angle(rad)", fontsize=18)
    # ax_yaw.legend(['Initial Actual', 'Initial Ref', 'NN Actual', 'NN Ref'], fontsize=12)
    ax_yaw.scatter(
        waypoints_time,
        np.zeros(len(waypoints)),
        c="k",
        marker="o",
        s=60,
        label="Waypoints",
    )

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")

    # close the figure
    plt.close(fig)


def plot_results_only_for_drag(
    sim_result_drag,
    sim_result_nn,
    waypoints,
    drag_comp_cost,
    predicted_cost,
    filename=None,
    waypoints_time=None,
):
    actual_yaw_drag = compute_yaw_from_quaternion(sim_result_drag["state"]["q"])
    actual_yaw_nn = compute_yaw_from_quaternion(sim_result_nn["state"]["q"])

    # Create the figure
    fig = plt.figure(figsize=(18, 8))

    # 3D Trajectory plot with waypoints
    ax_traj = fig.add_subplot(121, projection="3d")
    # Drag-aware trajectories
    ax_traj.plot3D(
        sim_result_drag["flat"]["x"][:, 0],
        sim_result_drag["flat"]["x"][:, 1],
        sim_result_drag["flat"]["x"][:, 2],
        color="lightgreen",
        linestyle="--",
        linewidth=4,
        label="minsnap+drag Reference",
    )
    ax_traj.plot3D(
        sim_result_drag["state"]["x"][:, 0],
        sim_result_drag["state"]["x"][:, 1],
        sim_result_drag["state"]["x"][:, 2],
        color="green",
        linestyle="-",
        linewidth=4,
        label="minsnap+drag Actual",
    )
    # Our method trajectories
    ax_traj.plot3D(
        sim_result_nn["flat"]["x"][:, 0],
        sim_result_nn["flat"]["x"][:, 1],
        sim_result_nn["flat"]["x"][:, 2],
        color="lightskyblue",
        linestyle="--",
        linewidth=4,
        label="Drag-aware Reference",
    )
    ax_traj.plot3D(
        sim_result_nn["state"]["x"][:, 0],
        sim_result_nn["state"]["x"][:, 1],
        sim_result_nn["state"]["x"][:, 2],
        color="blue",
        linestyle="-",
        linewidth=4,
        label="Drag-aware Actual",
    )

    # Waypoints
    ax_traj.scatter(
        waypoints[:, 0],
        waypoints[:, 1],
        waypoints[:, 2],
        c="k",
        marker="o",
        s=60,
        label="Waypoints",
    )

    ax_traj.set_title("3D Trajectories with Waypoints", fontsize=20)
    # adding meter as units to the axis
    ax_traj.set_xlabel("X(m)", fontsize=18)
    ax_traj.set_ylabel("Y(m)", fontsize=18)
    ax_traj.set_zlabel("Z(m)", fontsize=18)
    ax_traj.legend(fontsize=16)

    # # Set the view angle
    # ax_traj.view_init(elev=30, azim=210)   # Set the elevation and azimuth angles

    # print minsnap+drag Avg Cost from Sim
    print("minsnap+drag Cost from Sim: ", drag_comp_cost)
    print("Drag-aware Cost from Sim: ", predicted_cost)

    # Subplots for X, Y, Z, Yaw
    gs = fig.add_gridspec(4, 2)
    ax_x = fig.add_subplot(gs[0, 1])
    ax_y = fig.add_subplot(gs[1, 1])
    ax_z = fig.add_subplot(gs[2, 1])
    ax_yaw = fig.add_subplot(gs[3, 1])

    # Subplot for X
    ax_x.plot(
        sim_result_drag["time"],
        sim_result_drag["flat"]["x"][:, 0],
        color="lightgreen",
        linestyle="--",
        linewidth=4,
    )
    ax_x.plot(
        sim_result_drag["time"],
        sim_result_drag["state"]["x"][:, 0],
        color="green",
        linestyle="-",
        linewidth=4,
    )
    ax_x.plot(
        sim_result_nn["time"],
        sim_result_nn["flat"]["x"][:, 0],
        color="lightskyblue",
        linestyle="--",
        linewidth=4,
    )
    ax_x.plot(
        sim_result_nn["time"],
        sim_result_nn["state"]["x"][:, 0],
        color="blue",
        linestyle="-",
        linewidth=4,
    )
    ax_x.set_title("X Position Over Time", fontsize=20)
    ax_x.set_xlabel("Time(s)", fontsize=18)
    ax_x.set_ylabel("X Position(m)", fontsize=18)
    # ax_x.legend(['Initial Actual', 'Initial Ref', 'NN Actual', 'NN Ref'], fontsize=12)

    # Subplot for Y
    ax_y.plot(
        sim_result_drag["time"],
        sim_result_drag["flat"]["x"][:, 1],
        color="lightgreen",
        linestyle="--",
        linewidth=4,
    )
    ax_y.plot(
        sim_result_drag["time"],
        sim_result_drag["state"]["x"][:, 1],
        color="green",
        linestyle="-",
        linewidth=4,
    )
    ax_y.plot(
        sim_result_nn["time"],
        sim_result_nn["flat"]["x"][:, 1],
        color="lightskyblue",
        linestyle="--",
        linewidth=4,
    )
    ax_y.plot(
        sim_result_nn["time"],
        sim_result_nn["state"]["x"][:, 1],
        color="blue",
        linestyle="-",
        linewidth=4,
    )
    ax_y.set_title("Y Position Over Time", fontsize=20)
    ax_y.set_xlabel("Time(s)", fontsize=18)
    ax_y.set_ylabel("Y Position(m)", fontsize=18)
    # ax_y.legend(['Initial Actual', 'Initial Ref', 'NN Actual', 'NN Ref'], fontsize=12)

    # Subplot for Z
    ax_z.plot(
        sim_result_drag["time"],
        sim_result_drag["flat"]["x"][:, 2],
        color="lightgreen",
        linestyle="--",
        linewidth=4,
    )
    ax_z.plot(
        sim_result_drag["time"],
        sim_result_drag["state"]["x"][:, 2],
        color="green",
        linestyle="-",
        linewidth=4,
    )
    ax_z.plot(
        sim_result_nn["time"],
        sim_result_nn["flat"]["x"][:, 2],
        color="lightskyblue",
        linestyle="--",
        linewidth=4,
    )
    ax_z.plot(
        sim_result_nn["time"],
        sim_result_nn["state"]["x"][:, 2],
        color="blue",
        linestyle="-",
        linewidth=4,
    )
    ax_z.set_title("Z Position Over Time", fontsize=20)
    ax_z.set_xlabel("Time(s)", fontsize=18)
    ax_z.set_ylabel("Z Position(m)", fontsize=18)
    # ax_z.legend(['Initial Actual', 'Initial Ref', 'NN Actual', 'NN Ref'], fontsize=12)

    # Adding keyframes to the subplots
    for ax, dim in zip([ax_x, ax_y, ax_z], [0, 1, 2]):
        ax.scatter(
            waypoints_time,
            waypoints[:, dim],
            c="k",
            marker="o",
            s=60,
            label="Waypoints",
        )

    # Subplot for Yaw
    ax_yaw.plot(
        sim_result_drag["time"],
        actual_yaw_drag,
        color="green",
        linestyle="-",
        linewidth=4,
    )
    ax_yaw.plot(
        sim_result_drag["time"],
        sim_result_drag["flat"]["yaw"],
        color="lightgreen",
        linestyle="--",
        linewidth=4,
    )
    ax_yaw.plot(
        sim_result_nn["time"], actual_yaw_nn, color="blue", linestyle="-", linewidth=4
    )
    ax_yaw.plot(
        sim_result_nn["time"],
        sim_result_nn["flat"]["yaw"],
        color="lightskyblue",
        linestyle="--",
        linewidth=4,
    )

    ax_yaw.set_title("Yaw Angle Over Time", fontsize=20)
    ax_yaw.set_xlabel("Time(s)", fontsize=18)
    ax_yaw.set_ylabel("Yaw Angle(rad)", fontsize=18)
    # ax_yaw.legend(['Initial Actual', 'Initial Ref', 'NN Actual', 'NN Ref'], fontsize=12)
    ax_yaw.scatter(
        waypoints_time,
        np.zeros(len(waypoints)),
        c="k",
        marker="o",
        s=60,
        label="Waypoints",
    )

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")

    # close the figure
    plt.close(fig)


def plot_results(
    sim_result_init,
    sim_result_nn,
    waypoints,
    initial_cost,
    predicted_cost,
    filename=None,
    waypoints_time=None,
):
    actual_yaw_init = compute_yaw_from_quaternion(sim_result_init["state"]["q"])
    actual_yaw_nn = compute_yaw_from_quaternion(sim_result_nn["state"]["q"])

    # Create the figure
    fig = plt.figure(figsize=(18, 8))

    # 3D Trajectory plot with waypoints
    ax_traj = fig.add_subplot(121, projection="3d")
    # Minsnap trajectories
    ax_traj.plot3D(
        sim_result_init["flat"]["x"][:, 0],
        sim_result_init["flat"]["x"][:, 1],
        sim_result_init["flat"]["x"][:, 2],
        color="lightcoral",
        linestyle="--",
        linewidth=4,
        label="Minsnap Reference",
    )
    ax_traj.plot3D(
        sim_result_init["state"]["x"][:, 0],
        sim_result_init["state"]["x"][:, 1],
        sim_result_init["state"]["x"][:, 2],
        color="red",
        linestyle="-",
        linewidth=4,
        label="Minsnap Actual",
    )
    # Our method trajectories
    ax_traj.plot3D(
        sim_result_nn["flat"]["x"][:, 0],
        sim_result_nn["flat"]["x"][:, 1],
        sim_result_nn["flat"]["x"][:, 2],
        color="lightskyblue",
        linestyle="--",
        linewidth=4,
        label="Drag-aware Reference",
    )
    ax_traj.plot3D(
        sim_result_nn["state"]["x"][:, 0],
        sim_result_nn["state"]["x"][:, 1],
        sim_result_nn["state"]["x"][:, 2],
        color="blue",
        linestyle="-",
        linewidth=4,
        label="Drag-aware Actual",
    )
    # # set x, y, z limits
    # ax_traj.set_xlim(-3, 3)
    # ax_traj.set_ylim(-3, 3)
    # ax_traj.set_zlim(-3, 3)
    # Waypoints
    ax_traj.scatter(
        waypoints[:, 0],
        waypoints[:, 1],
        waypoints[:, 2],
        c="k",
        marker="o",
        s=60,
        label="Waypoints",
    )

    ax_traj.set_title("3D Trajectories with Waypoints", fontsize=20)
    ax_traj.set_xlabel("X(m)", fontsize=18)
    ax_traj.set_ylabel("Y(m)", fontsize=18)
    ax_traj.set_zlabel("Z(m)", fontsize=18)
    ax_traj.legend(fontsize=16)

    # # Set the view angle
    # ax_traj.view_init(elev=30, azim=210)   # Set the elevation and azimuth angles

    # print Minsnap Avg Cost from Sim:and Drag-aware Avg Cost from Sim:
    print("Minsnap Cost from Sim: ", initial_cost)
    print("Drag-aware Cost from Sim: ", predicted_cost)

    # Subplots for X, Y, Z, Yaw
    gs = fig.add_gridspec(4, 2)
    ax_x = fig.add_subplot(gs[0, 1])
    ax_y = fig.add_subplot(gs[1, 1])
    ax_z = fig.add_subplot(gs[2, 1])
    ax_yaw = fig.add_subplot(gs[3, 1])

    # Subplot for X
    ax_x.plot(
        sim_result_init["time"],
        sim_result_init["flat"]["x"][:, 0],
        color="lightcoral",
        linestyle="--",
        linewidth=4,
    )
    ax_x.plot(
        sim_result_init["time"],
        sim_result_init["state"]["x"][:, 0],
        color="red",
        linestyle="-",
        linewidth=4,
    )
    ax_x.plot(
        sim_result_nn["time"],
        sim_result_nn["flat"]["x"][:, 0],
        color="lightskyblue",
        linestyle="--",
        linewidth=4,
    )
    ax_x.plot(
        sim_result_nn["time"],
        sim_result_nn["state"]["x"][:, 0],
        color="blue",
        linestyle="-",
        linewidth=4,
    )
    ax_x.set_title("X Position Over Time", fontsize=20)
    ax_x.set_xlabel("Time(s)", fontsize=18)
    ax_x.set_ylabel("X Position(m)", fontsize=18)
    # ax_x.legend(['Initial Actual', 'Initial Ref', 'NN Actual', 'NN Ref'], fontsize=12)

    # Subplot for Y
    ax_y.plot(
        sim_result_init["time"],
        sim_result_init["flat"]["x"][:, 1],
        color="lightcoral",
        linestyle="--",
        linewidth=4,
    )
    ax_y.plot(
        sim_result_init["time"],
        sim_result_init["state"]["x"][:, 1],
        color="red",
        linestyle="-",
        linewidth=4,
    )
    ax_y.plot(
        sim_result_nn["time"],
        sim_result_nn["flat"]["x"][:, 1],
        color="lightskyblue",
        linestyle="--",
        linewidth=4,
    )
    ax_y.plot(
        sim_result_nn["time"],
        sim_result_nn["state"]["x"][:, 1],
        color="blue",
        linestyle="-",
        linewidth=4,
    )
    ax_y.set_title("Y Position Over Time", fontsize=20)
    ax_y.set_xlabel("Time(s)", fontsize=18)
    ax_y.set_ylabel("Y Position(m)", fontsize=18)
    # ax_y.legend(['Initial Actual', 'Initial Ref', 'NN Actual', 'NN Ref'], fontsize=12)

    # Subplot for Z
    ax_z.plot(
        sim_result_init["time"],
        sim_result_init["flat"]["x"][:, 2],
        color="lightcoral",
        linestyle="--",
        linewidth=4,
    )
    ax_z.plot(
        sim_result_init["time"],
        sim_result_init["state"]["x"][:, 2],
        color="red",
        linestyle="-",
        linewidth=4,
    )
    ax_z.plot(
        sim_result_nn["time"],
        sim_result_nn["flat"]["x"][:, 2],
        color="lightskyblue",
        linestyle="--",
        linewidth=4,
    )
    ax_z.plot(
        sim_result_nn["time"],
        sim_result_nn["state"]["x"][:, 2],
        color="blue",
        linestyle="-",
        linewidth=4,
    )
    ax_z.set_title("Z Position Over Time", fontsize=20)
    ax_z.set_xlabel("Time(s)", fontsize=18)
    ax_z.set_ylabel("Z Position(m)", fontsize=18)
    # ax_z.legend(['Initial Actual', 'Initial Ref', 'NN Actual', 'NN Ref'], fontsize=12)

    # Adding keyframes to the subplots
    for ax, dim in zip([ax_x, ax_y, ax_z], [0, 1, 2]):
        ax.scatter(
            waypoints_time,
            waypoints[:, dim],
            c="k",
            marker="o",
            s=60,
            label="Waypoints",
        )

    # Subplot for Yaw
    ax_yaw.plot(
        sim_result_init["time"],
        actual_yaw_init,
        color="red",
        linestyle="-",
        linewidth=4,
    )
    ax_yaw.plot(
        sim_result_init["time"],
        sim_result_init["flat"]["yaw"],
        color="lightcoral",
        linestyle="--",
        linewidth=4,
    )
    ax_yaw.plot(
        sim_result_nn["time"], actual_yaw_nn, color="blue", linestyle="-", linewidth=4
    )
    ax_yaw.plot(
        sim_result_nn["time"],
        sim_result_nn["flat"]["yaw"],
        color="lightskyblue",
        linestyle="--",
        linewidth=4,
    )
    ax_yaw.set_title("Yaw Angle Over Time", fontsize=20)
    ax_yaw.set_xlabel("Time(s)", fontsize=18)
    ax_yaw.set_ylabel("Yaw Angle(rad)", fontsize=18)
    # ax_yaw.legend(['Initial Actual', 'Initial Ref', 'NN Actual', 'NN Ref'], fontsize=12)
    ax_yaw.scatter(
        waypoints_time,
        np.zeros(len(waypoints)),
        c="k",
        marker="o",
        s=60,
        label="Waypoints",
    )

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")

    # close the figure
    plt.close(fig)


def main():
    # Initialize neural network
    rho = 0
    drag_coeff = 3
    input_size = 96  # number of coeff

    with open(
        # r"/home/user/code/quadrotor-drag-exp/AeroWrenchPlanner/learning/params.yaml"
        # r"/workspace/AeroWrenchPlanner/learning/params.yaml"
        r"/home/sanusha/codes/Quadrotor-planning-via-control-decomposition/configs/params.yaml"
    ) as f:
        yaml_data = yaml.load(f, Loader=yaml.RoundTripLoader)

    num_hidden = yaml_data["num_hidden"]
    batch_size = yaml_data["batch_size"]
    learning_rate = yaml_data["learning_rate"]
    num_epochs = yaml_data["num_epochs"]
    model_save = yaml_data["save_path"] + str(rho) + str(drag_coeff)

    # with open(r"/workspace/rotorpy/learning/params.yaml") as f:
    # with open(r"/home/user/code/quadrotor-drag-exp/AeroWrenchPlanner/learning/params.yaml") as f:
    #     yaml_data = yaml.load(f, Loader=yaml.RoundTripLoader)

    # num_hidden = yaml_data["num_hidden"]

    # Path to your CSV file
    # csv_file_path = "/workspace/data_output/data_diff_rho.csv"
    csv_file_path = "/home/sanusha/codes/Quadrotor-planning-via-control-decomposition/data/data_diff_rho_drag" + str(drag_coeff) + ".csv"
    # csv_file_path = cwd + "/../../data/data_diff_rho_drag" + str(drag_coeff) + ".csv"
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

    trained_model = model.bind(trained_model_state.params)

    vf = (trained_model, trained_model_state.params)

    # Load the trained model
    # model = MLP(num_hidden=num_hidden, num_outputs=1)
    # # Printing the model shows its attributes
    # print(model)

    # rng = jax.random.PRNGKey(427)
    # rng, inp_rng, init_rng = jax.random.split(rng, 3)

    # # Initialize the model
    # # params = model.init(init_rng, inp)

    # # optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)

    # model_save = yaml_data["save_path"] + str(rho) + str(drag_coeff)
    # print("model_save", model_save)

    # trained_model_state = flax.core.freeze(restore_checkpoint(None, model_save, 7))
    # vf = (model, trained_model_state["params"])

    # Define the quadrotor parameters
    world_size = 10
    num_waypoints = 4
    vavg = 2
    random_yaw = False
    yaw_min = -0.85 * np.pi
    yaw_max = 0.85 * np.pi

    world_buffer = 2
    min_distance = 1
    max_distance = min_distance + 3
    start_waypoint = None  # If you want to start at a specific waypoint, specify it using [xstart, ystart, zstart]
    end_waypoint = None  # If you want to end at a specific waypoint, specify it using [xend, yend, zend]

    # Now create the world, vehicle, and controller objects.
    world = World.empty(
        [
            -world_size / 2,
            world_size / 2,
            -world_size / 2,
            world_size / 2,
            -world_size / 2,
            world_size / 2,
        ]
    )
    vehicle = Multirotor(quad_params)
    controller = SE3Control(quad_params)

    cost_differences = []

    i = 88

    waypoints = sample_waypoints(
        num_waypoints=num_waypoints,
        world=world,
        world_buffer=world_buffer,
        min_distance=min_distance,
        max_distance=max_distance,
        start_waypoint=start_waypoint,
        end_waypoint=end_waypoint,
        rng=None,
        seed=i,
    )

    print("waypoints", waypoints)

    yaw_angles_zero = np.zeros(len(waypoints))

    # /workspace/rotorpy/rotorpy/sim_figures/
    # figure_path = "/workspace/data_output/sim_figures_with_rho" + str(rho)
    # figure_path = "/home/user/code/quadrotor-drag-exp/sim_figures_with_rho" + str(rho) 
    # figure_path = "/home/user/code/quadrotor-drag-exp/sim_figures_with_rho" + str(rho) + "jaxopt"
    figure_path = "/home/sanusha/codes/Quadrotor-planning-via-control-decomposition/scripts/plots_inf"
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)

    start_time = time.time()
    total_time = 0

    # run simulation and compute cost for the minsnap trajectory with l1 adaptive ctrl
    # controller_l1adapt = L1SE3Control(quad_params, adaptive_alpha=0.3)
    # (
    #     sim_result_l1,
    #     trajectory_cost_l1,
    #     _,
    #     _,
    #     summary_result_l1,
    # ) = run_simulation_and_compute_cost(
    #     waypoints,
    #     yaw_angles_zero,
    #     vavg,
    #     use_neural_network=False,
    #     regularizer=None,
    #     vehicle=vehicle,
    #     controller=controller_l1adapt,
    #     robust_c=rho,
    # )

    # run simulation and compute cost for the minsnap trajectory with drag compensation
    # controller_with_drag_compensation = SE3Control(quad_params, drag_compensation=True)
    # (
    #     sim_result_drag,
    #     trajectory_cost_drag,
    #     _,
    #     _,
    #     summary_result_drag,
    # ) = run_simulation_and_compute_cost(
    #     waypoints,
    #     yaw_angles_zero,
    #     vavg,
    #     use_neural_network=False,
    #     regularizer=None,
    #     vehicle=vehicle,
    #     controller=controller_with_drag_compensation,
    #     robust_c=rho,
    # )
    # run simulation and compute cost for the initial trajectory
    (
        sim_result_init,
        trajectory_cost_init,
        waypoints_time,
        _,
        summary_result_init,
    ) = run_simulation_and_compute_cost(
        waypoints,
        yaw_angles_zero,
        vavg,
        use_neural_network=False,
        regularizer=None,
        vehicle=vehicle,
        controller=controller,
        robust_c=rho,
    )
    # write_to_csv(figure_path + "/summary_data_init_header_v2_new65.csv", summary_result_init)
    # write_to_csv(output_csv_file, result)
    # run simulation and compute cost for the modified trajectory
    (
        sim_result_nn,
        trajectory_cost_nn,
        _,
        nan_encountered,
        summary_result_nn,
    ) = run_simulation_and_compute_cost(
        waypoints,
        yaw_angles_zero,
        vavg,
        use_neural_network=True,
        regularizer=vf,
        vehicle=vehicle,
        controller=controller,
        robust_c=rho,
    )
    # write_to_csv(figure_path + "/summary_data_nn_header_v2_new65.csv", summary_result_nn)
    print("nan_encountered in inference", nan_encountered)
    if nan_encountered == False:
        print(f"Trajectory {i} initial cost: {trajectory_cost_init}")
        print(f"Trajectory {i} neural network modified cost: {trajectory_cost_nn}")
        print(f"Trajectory {i} drag comp ctrl cost: {trajectory_cost_drag}")
        print(f"Trajectory {i} l1 adaptive ctrl cost: {trajectory_cost_l1}")
        cost_diff = trajectory_cost_nn - trajectory_cost_init
        cost_differences.append((trajectory_cost_init, trajectory_cost_nn, cost_diff))
        # plot_results_only_for_drag(
        #     sim_result_l1,
        #     sim_result_nn,
        #     waypoints,
        #     trajectory_cost_l1,
        #     trajectory_cost_nn,
        #     # filename=figure_path + f"/sum_cost_3Dtrajectory_only_dragcomp_{i}_.png",
        #     filename=figure_path + f"/jaxopt_sum_cost_3Dtrajectory_only_l1_{i}_.png",
        #     waypoints_time=waypoints_time,
        # )
        # plot_results_only_for_drag(
        #     sim_result_drag,
        #     sim_result_nn,
        #     waypoints,
        #     trajectory_cost_drag,
        #     trajectory_cost_nn,
        #     # filename=figure_path + f"/sum_cost_3Dtrajectory_only_dragcomp_{i}_.png",
        #     filename=figure_path + f"/jaxopt_sum_cost_3Dtrajectory_only_dragcomp_{i}_.png",
        #     waypoints_time=waypoints_time,
        # )
        # plot_results_with_drag(sim_result_init, sim_result_nn, sim_result_drag, waypoints, trajectory_cost_init, trajectory_cost_nn, trajectory_cost_drag, filename=figure_path + f"/sum_cost_3Dtrajectory_with_dragcomp_{i}.png", waypoints_time=waypoints_time)
        plot_results(
            sim_result_init,
            sim_result_nn,
            waypoints,
            trajectory_cost_init,
            trajectory_cost_nn,
            # filename=figure_path + f"/sum_cost_3Dtrajectory_{i}_.png",
            filename=figure_path + f"/jaxopt_sum_cost_3Dtrajectory_{i}_.png",
            waypoints_time=waypoints_time,
        )
        # plot_cumulative_tracking_error(
        #     sim_result_init,
        #     sim_result_nn,
        #     sim_result_drag,
        #     sim_result_l1,
        #     # filename=figure_path + f"/cumulative_tracking_error_{i}_.png",
        #     filename=figure_path + f"/jaxopt_cumulative_tracking_error_{i}_.png",
        # )

    end_time = time.time()
    elapsed_time = end_time - start_time
    total_time += elapsed_time
    print(f"Elapsed time for trajectory {i}: {elapsed_time} seconds")

    # # Save the cost data to a CSV file
    # costs_df = pd.DataFrame(cost_differences, columns=['Initial Cost', 'NN Modified Cost', 'Cost Difference'])
    # # save to figure path
    # costs_df.to_csv(figure_path + "/cost_data.csv", index=False)


if __name__ == "__main__":
    # try:
    #     main()
    # except rospy.ROSInterruptException:
    #     pass
    
    main()