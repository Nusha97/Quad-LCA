# Quad-LCA: A Layered Control Architecture for Quadrotor Planning

This repository contains the implementation of the methods described from "[A Data-Driven Approach to Synthesizing Dynamics-Aware Trajectories for Underactuated Robotic Systems](https://arxiv.org/abs/2307.13782)" and its extension specializing the contributions to a Crazyflie 2.0 "[Why Change Your Controller When You Can Change Your Planner: Drag-Aware Trajectory Generation for Quadrotor Systems](https://arxiv.org/abs/2401.04960)". In the original methods paper, we showed how one can derive a dynamics-aware trajectory generation problem for motion planning and use value iteration to approximately learn an objective function denoting deviation of the optimized reference trajectory and system executed trajectory. In this repository, we provide a streamlined API for data generation, training and evaluation. We leverage RotorPy as our simulator backbone to build experiments for testing across various mass, inertia and drag coefficient configurations.

## Overview

Path planners in robotics often separate trajectory planning and feedback control, which can lead to suboptimal performance due to mismatches between planned trajectories and the system's actuator capabilities. This work proposes a decomposition that obtains trajectory generation with a tracking penalty regularizer, learned from system rollouts, to ensure actuator feasibility.

Key contributions include:

- **Penalty Method Reformulation**: Decomposing the global nonlinear optimal control problem into trajectory planning and feedback control layers.
- **Tracking Penalty Regularizer**: Introducing a regularizer that encodes the dynamic feasibility of generated trajectories, learned from system rollouts.
- **Empirical Validation**: Demonstrating significant improvements in tracking feasibility through simulations with a unicycle model and both simulations and hardware experiments with two quadrotor platforms namely, a Crazyflie and Dragonfly.

## Repository Structure

- `configs/`: Contains yaml files for different experiment configurations and parameters for neural network.
- `notebooks/`: Includes Python notebooks to help visualize and debug different components of the code.
- `scripts/`: Contains Python scripts for 
- `models/`: Pre-trained models and parameters for the unicycle and quadrotor systems.
- `src/`: 
- `Dockerfile`: Provides the dependencies to setup a container and an environment to support data generation, training and inference in simulation
- `build.sh`: Builds the docker container
- `run.sh`: Executes and run the docker container

## Installation

To set up the environment, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone git@github.com:Nusha97/Quad-LCA.git
   cd Quad-LCA
   ```

2. **Build and run docker container** (optional but recommended):
   ```bash
   ./build.sh
   ./run.sh
   ```

3. **Activate conda environment**:
   ```bash
   conda activate lca_sanusha # TODO: change this to lca
   ```

## Training and inference in simulation

1. **Generate data**:
   ```bash
   python scripts/rotorpy_data.py # TODO: Add the experiment configs and parsing tools
   ```

2. **Train a model**:
   ```bash
   python scripts/training.py # TODO: Add parser to select between MLPs and ICNNs
   ```

3. **Run inference on a trained model**:
   ```bash
   python scripts/singletraj_inference.py # TODO: Add options to pass in waypoints or initialize coefficients
   ```

## Deploying in ROS simulation

1. **Run rviz simulator**
   ```bash
   ./demo_sim.sh 1 # TODO: This needs a submodule to access roscd kr_multimav_manager/scripts 
   ```

2. **Deploy inference model for evaluation**
   ```bash
   python src/scripts/inference/regularized_trajectory.py # TODO: cleanup since this currently only supports RotorPy
   ```

3. **To deploy on hardware**
### To sync messages and write bag in sim
`rosrun layered_ref_control sync_msg _sim:=true _namespace:=dragonfly1`

  * Bag file gets written to home folder with current data/time as filename

### To sync messages and write bag hw, set the correct namespace based on the MAV ID
`rosrun layered_ref_control sync_msg _sim:=false _namespace:=dragonfly25`

## Citing This Work

If you find this work useful in your research, please consider citing:

```bibtex
@article{srikanthan2023dynamicsaware,
  title={A Data-Driven Approach to Synthesizing Dynamics-Aware Trajectories for Underactuated Robotic Systems},
  author={Srikanthan, Anusha and Yang, Fengjun and Spasojevic, Igor and Thakur, Dinesh and Kumar, Vijay and Matni, Nikolai},
  journal={arXiv preprint arXiv:2307.13782},
  year={2023}
}

@article{zhang2024change,
  title={Why change your controller when you can change your planner: Drag-aware trajectory generation for quadrotor systems},
  author={Zhang, Hanli and Srikanthan, Anusha and Folk, Spencer and Kumar, Vijay and Matni, Nikolai},
  journal={arXiv preprint arXiv:2401.04960},
  year={2024}
}
```

## Acknowledgements

This research is supported in part by NSF award CPS-2038873, NSF CAREER award ECCS-2045834, NSF Grant CCR-2112665.

For any questions or issues, please open an issue on this repository or contact the authors directly.