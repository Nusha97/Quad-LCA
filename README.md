# A ros package for layered control architectures applied to quadrotor trajectory generation and control

This repository contains the implementation of the methods described in the paper "[A Data-Driven Approach to Synthesizing Dynamics-Aware Trajectories for Underactuated Robotic Systems](https://arxiv.org/abs/2307.13782)" by Anusha Srikanthan et al. In this paper, we showed how one can derive a dynamics-aware trajectory generation problem for motion planning and use value iteration to approximately learn an objective function denoting deviation of the optimized reference trajectory and system executed trajectory.

## Overview

Traditional layered control architectures in robotics often separate trajectory planning and feedback control, which can lead to suboptimal performance due to mismatches between planned trajectories and the system's dynamic capabilities. This work proposes an alternative approach that integrates trajectory generation with a tracking penalty regularizer, learned from system rollouts, to ensure dynamic feasibility.

Key contributions include:

- **Augmented Lagrangian Reformulation**: Decomposing the global nonlinear optimal control problem into trajectory planning and feedback control layers.
- **Tracking Penalty Regularizer**: Introducing a regularizer that encodes the dynamic feasibility of generated trajectories, learned from system rollouts.
- **Empirical Validation**: Demonstrating significant improvements in computation time and dynamic feasibility through simulations with a unicycle model and both simulations and hardware experiments with a quadrotor platform.

## Repository Structure

- `src/`: Contains the source code for Python-based ROS interface.
- `scripts/`: Contains the source code for the neural network architecture, the learned tracking penalty regularizer, the online planner.
- `configs/`: Contains the configuration parameters for both the unicycle and the quadrotor experiments.
- `data/`: Includes datasets from system rollouts used for learning the tracking penalty regularizer.
- `notebooks/': Includes Python notebooks to help visualize and debug different components of the code.
- `models/`: Pre-trained models and parameters for the unicycle and quadrotor systems.
- `experiments/`: Scripts and configurations to reproduce the experiments presented in the paper.
- `docs/`: Documentation and supplementary materials.

## Installation

To set up the environment, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone git@github.com:Nusha97/Quadrotor-planning-via-control-decomposition.git
   cd Quadrotor-planning-via-control-decomposition
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Citing This Work

If you find this work useful in your research, please consider citing:

```bibtex
@article{srikanthan2023dynamicsaware,
  title={A Data-Driven Approach to Synthesizing Dynamics-Aware Trajectories for Underactuated Robotic Systems},
  author={Srikanthan, Anusha and Yang, Fengjun and Spasojevic, Igor and Thakur, Dinesh and Kumar, Vijay and Matni, Nikolai},
  journal={arXiv preprint arXiv:2307.13782},
  year={2023}
}
```

## Acknowledgements

This research is supported in part by NSF award CPS-2038873, NSF CAREER award ECCS-2045834, NSF Grant CCR-2112665, and a Google Research Scholar award.

For any questions or issues, please open an issue on this repository or contact the authors directly.

### To sync messages and write bag in sim
`rosrun layered_ref_control sync_msg _sim:=true _namespace:=dragonfly1`

  * Bag file gets written to home folder with current data/time as filename

### To sync messages and write bag hw, set the correct namespace based on the MAV ID
`rosrun layered_ref_control sync_msg _sim:=false _namespace:=dragonfly25`
