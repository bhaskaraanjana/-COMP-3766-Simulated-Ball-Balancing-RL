# -COMP-3766-Simulated-Ball-Balancing-RL# Tilting Table Ball Balancer using Reinforcement Learning (PPO)

## Project Overview

This project implements a system for balancing a rolling ball on a tilting table within the Gazebo physics simulator, controlled by a Reinforcement Learning (RL) agent trained using Proximal Policy Optimization (PPO). The system integrates ROS Noetic, Gazebo, Gymnasium (formerly Gym), and Stable Baselines3.

The primary goal is to train an agent that learns to manipulate the table's two rotational joints (`joint_x`, `joint_y`) based on observations of the ball's state (and potentially the table's state) to keep the ball centered on the plate for as long as possible.

This project was developed as the final project for **COMP3766 Introduction to Robotic Manipulation** at Memorial University of Newfoundland, instructed by Dr. Vinicius Prado da Fonseca

## Features

* **`table_env_pkg`**: The main ROS package containing the environment, models, and scripts.
* **Gazebo Simulation**: Simulates the tilting table and the rolling ball physics.
    * Models: `tilting_table` and `rolling_ball` define the visual and physical properties.
    * `universal_joint_controller.cpp`: A custom Gazebo plugin that controls the table's two joints (`joint_x`, `joint_y`) based on effort commands received via a ROS topic (`/tilting_table/tilt_cmd`).
* **RL Environment (`tilting_table_env.py`)**: Defines the `gymnasium` environment, including:
    * **Observation Space**: Relative position (x, y) and velocity (vx, vy) of the ball with respect to the table plate (4 dimensions). *Note: An 8-dimensional version (`tilting_table_env(8d obs).py`) also exists, potentially including joint angles and velocities.*
    * **Action Space**: Continuous effort values for the two table joints (`joint_x`, `joint_y`).
    * **Reward Function**: Rewards the agent for keeping the ball near the center of the table and penalizes large actions.
    * **Episode Termination**: Ends if the ball falls off the table (checked by Z-height or exceeding X/Y boundaries) or reaches the maximum step count.
* **Training Script (`train_ppo.py`)**: Uses `stable-baselines3` to train a PPO agent within the `TiltingTableEnv`. It includes callbacks for saving checkpoints and evaluating the best model.
* **Testing Script (`test_ppo.py`)**: Loads a pre-trained PPO model and evaluates its performance in the environment.
* **Utility Script (`ball_tracker.py`)**: Subscribes to Gazebo model states (`/gazebo/model_states`) to calculate and log the ball's position relative to the table plate.
* TensorBoard integration for monitoring training progress.

## Requirements / Dependencies

### ROS (Noetic)

Ensure you have a working ROS Noetic installation with Gazebo 11.

* `catkin`
* `roscpp`
* `rospy` 
* `std_msgs` 
* `geometry_msgs` 
* `sensor_msgs` 
* `std_srvs` 
* `gazebo_ros`
* `gazebo_msgs`
* `gazebo_ros_control`
* `controller_manager`
* `effort_controllers`
* `joint_state_controller`
* `robot_state_publisher`
* `xacro`
* `tf`

* Ubuntu 20.04 (or compatible)
* ROS Noetic Ninjemys (including `gazebo_ros_pkgs`)
* Gazebo (version compatible with ROS Noetic)
* Python 3.8+
* Catkin Tools (`catkin_make` or `catkin build`)
  
### Python

See `requirements.txt` for specific versions. Key libraries include:

* `stable-baselines3`
* `gymnasium`
* `numpy`
* `torch` (as a dependency of stable-baselines3)
* 
## Installation & Setup

1.  **Clone the Repository:** Clone this repository into the `src` directory of your Catkin workspace.
    ```bash
    cd ~/catkin_ws/
    git clone https://github.com/bhaskaraanjana/-COMP-3766-Simulated-Ball-Balancing-RL.git
    ```
2.  **Install Python Dependencies:** (Recommended: Use a Python virtual environment)
    ```bash
    cd ~/catkin_ws/
    # Optional: Create and activate virtual environment
    # python3 -m venv venv
    # source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Build the Catkin Workspace:**
    ```bash
    cd ~/catkin_ws
    catkin_make # or catkin build
    ```
4.  **Source the Workspace:**
    ```bash
    source ~/catkin_ws/devel/setup.bash
    # Add this line to your ~/.bashrc for convenience
    ```

## Usage

1.  **Launch Simulation:** Open a terminal, source your workspace, and launch the Gazebo world with the table and ball. (You'll need to create a launch file, e.g., `balancing_world.launch`, for this).
    ```bash
    source ~/catkin_ws/devel/setup.bash
    roslaunch table_env_pkg start_sim.launch 
    ```
2.  **Train a New Agent:** Open another terminal, source the workspace (and Python venv if used).
    ```bash
    # Navigate to the scripts directory
    source ~/catkin_ws/devel/setup.bash
    cd ~/catkin_ws/src/table_env_pkg/scripts/
    python3 train_ppo.py
    ```
    * Monitor training progress using TensorBoard: `tensorboard --logdir ~/catkin_ws/training_logs/`
3.  **Test a Trained Agent:** Open another terminal, source the workspace (and Python venv if used). Make sure to put the trained model in ~/catkin_ws/trained_models/
    ```bash
    # Navigate to the scripts directory
    source ~/catkin_ws/devel/setup.bash
    cd ~/catkin_ws/src/table_env_pkg/scripts/
    python3 test_ppo.py
    ```

## Future Work/Areas for contribution
**Enhanced Observation Space:** Implementing and training with the proposed 8D observation space (adding table joint angles/velocities) could improve policy robustness to initial table tilt, addressing a potential limitation of the current 4D space.

**Systematic Hyperparameter Optimization:** Employing automated tuning tools (like Optuna or Ray Tune) for the PPO hyperparameters could potentially yield improved performance or faster convergence compared to the iterative manual tuning performed.

**Reward Function Refinement:** Further experimentation could optimize the reward function, either by fine-tuning the existing centering and action penalty constants or by investigating alternative structures, such as adding penalties for high ball velocity near the center.

**Advanced Reset Mechanisms:** Exploring the use of ros_control with position controllers could provide a more standard and potentially faster method for resetting joint angles precisely to zero, eliminating the need for physical settling time.

**Robustness Testing:** Conducting more extensive testing by evaluating the agent's performance across a wider range of initial conditions and its ability to handle simulated external disturbances would provide a more comprehensive assessment of its capabilities.

