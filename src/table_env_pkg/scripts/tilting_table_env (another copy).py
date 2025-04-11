#!/usr/bin/env python3
# Imports for check_env testing
from stable_baselines3.common.env_checker import check_env
import traceback # Import traceback to print full errors

# Core environment imports
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rospy
import sys
import time  # For potential sleeps

# ROS Messages and Services
from gazebo_msgs.msg import LinkStates, ModelState
from gazebo_msgs.srv import ApplyJointEffort, ApplyJointEffortRequest
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Wrench

# Helper for TF transformations
import tf.transformations as tf_trans

# --- Helper Function ---
def calculate_relative_position(ball_world_pos, plate_world_pose):
    """ Calculates ball position relative to the plate frame. """
    plate_pos = np.array([plate_world_pose.position.x, plate_world_pose.position.y, plate_world_pose.position.z])
    plate_orient_q = np.array([plate_world_pose.orientation.x, plate_world_pose.orientation.y,
                            plate_world_pose.orientation.z, plate_world_pose.orientation.w])
    ball_pos = np.array([ball_world_pos.x, ball_world_pos.y, ball_world_pos.z])

    diff_world = ball_pos - plate_pos

    # Ensure quaternion is valid and normalized
    if np.sum(np.square(plate_orient_q)) < 1e-6:
        plate_orient_q = np.array([0.0, 0.0, 0.0, 1.0])
        # rospy.logwarn_throttle(5.0,"Plate orientation quaternion is zero or near-zero, using identity.") # Removed debug
    else:
        plate_orient_q = plate_orient_q / np.linalg.norm(plate_orient_q)

    try:
        plate_orient_q_inv = tf_trans.quaternion_inverse(plate_orient_q)
        diff_quat = np.append([0], diff_world)
        diff_plate_frame_quat = tf_trans.quaternion_multiply(
            tf_trans.quaternion_multiply(plate_orient_q_inv, diff_quat),
            plate_orient_q
        )
        relative_pos = diff_plate_frame_quat[1:]
    except (np.linalg.LinAlgError, ValueError) as e:
         rospy.logwarn_throttle(5.0,f"Quaternion math error for position: {e}. Using world difference.")
         relative_pos = diff_world

    return relative_pos

# --- Environment Class ---
class TiltingTableEnv(gym.Env):
    metadata = {'render_modes': ['human'], "render_fps": 60}
    SURVIVAL_BONUS = 0.05
    TERMINATION_PENALTY = -5.0
    BOUND_LIMIT = 0.21

    MAX_CENTERING_REWARD = 2.0  # Max reward per step at center (tune this)
    CENTERING_REWARD_SCALE = 5.0 # Sharpness of reward fall-off (tune this)
    Z_THRESHOLD = 0.08 # Use lower threshold

    MAX_STEPS_PER_EPISODE = 1000
    ACTION_PENALTY_WEIGHT = 0.2 # Use increased penalty weight from previous suggestion

    def __init__(self):
        super(TiltingTableEnv, self).__init__()

        max_effort = 5.0  # Keep reduced max effort
        self.action_space = spaces.Box(low=-max_effort, high=max_effort, shape=(2,), dtype=np.float32)
        obs_limit = np.array([0.2, 0.2, 5.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_limit, high=obs_limit, shape=(4,), dtype=np.float32)

        # Initialize Gym's random number generator for seeding and randomness in resets
        self.np_random, _ = gym.utils.seeding.np_random()
        # Set a configurable ball drop height (adjust as needed)
        self.ball_drop_height = 0.6

        # Initialize the ROS node if not already initialized
        try:
            if not rospy.core.is_initialized():
                rospy.init_node('tilting_table_env', anonymous=True)
                self.node_initialized_by_env = True
                rospy.loginfo("ROS node initialized by TiltingTableEnv.")
            else:
                self.node_initialized_by_env = False
                rospy.logdebug("ROS node already initialized externally.")
        except rospy.exceptions.ROSException as e:
             rospy.logwarn(f"Error checking ROS initialization: {e}")
             self.node_initialized_by_env = False

        rospy.loginfo("Waiting for Gazebo services...")
        try:
            rospy.wait_for_service('/gazebo/apply_joint_effort', timeout=5.0)
            rospy.wait_for_service('/gazebo/set_model_state', timeout=5.0)
            rospy.wait_for_service('/gazebo/pause_physics', timeout=5.0)
            rospy.wait_for_service('/gazebo/unpause_physics', timeout=5.0)
            rospy.wait_for_service('/gazebo/reset_simulation', timeout=5.0)  # Added reset service
            self.apply_effort_proxy = rospy.ServiceProxy('/gazebo/apply_joint_effort', ApplyJointEffort)
            self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
            self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
            self.reset_simulation_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)  # New proxy for full reset
            rospy.loginfo("Gazebo services connected.")
        except rospy.exceptions.ROSException as e:
             rospy.logerr(f"Failed to connect to Gazebo services: {e}")
             sys.exit()

        self.latest_link_states = None
        rospy.Subscriber('/gazebo/link_states', LinkStates, self._link_states_callback)
        rospy.loginfo("Subscribed to /gazebo/link_states")

        self.current_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.plate_pose_world = None
        self.rate = rospy.Rate(100)
        self.step_count = 0

        rospy.loginfo("Waiting for initial link state message...")
        start_wait_time = rospy.get_time()
        while self.latest_link_states is None and not rospy.is_shutdown():
             if rospy.get_time() - start_wait_time > 10.0:
                 rospy.logerr("Timeout waiting for initial /gazebo/link_states message.")
                 sys.exit()
             try:
                 self.rate.sleep()
             except rospy.exceptions.ROSInterruptException:
                 rospy.loginfo("Shutdown requested.")
                 return
        rospy.loginfo("Initial link state received.")
        self._process_latest_link_states()
        # Process initial state to populate observation
        if not self._process_latest_link_states():
             rospy.logerr("Failed to process initial link states after receiving message.")
             # Handle error: maybe retry or exit


    def _link_states_callback(self, msg):
        """ Stores the latest LinkStates message. """
        self.latest_link_states = msg


    def _process_latest_link_states(self):
        """ Processes the stored LinkStates msg to update observation, with detailed checks. """
        if self.latest_link_states is None:
            rospy.logwarn_throttle(5.0,"Cannot process link states, none received yet.")
            self.current_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            return False

        msg = self.latest_link_states
        try:
            ball_link_name = 'ball_1::ball_link'
            plate_link_name = 'table_1::plate'

            if ball_link_name in msg.name and plate_link_name in msg.name:
                ball_index = msg.name.index(ball_link_name)
                plate_index = msg.name.index(plate_link_name)

                ball_pose_world = msg.pose[ball_index]
                ball_twist_world = msg.twist[ball_index]
                self.plate_pose_world = msg.pose[plate_index]

                # Check Raw Data Validity
                raw_poses_valid = np.all(np.isfinite([ball_pose_world.position.x, ball_pose_world.position.y, ball_pose_world.position.z,
                                                     self.plate_pose_world.position.x, self.plate_pose_world.position.y, self.plate_pose_world.position.z]))
                if not raw_poses_valid:
                     rospy.logerr("Non-finite value found in raw link positions! Skipping state update.")
                     return False # Keep previous observation by returning False
                raw_orient_valid = np.all(np.isfinite([self.plate_pose_world.orientation.x, self.plate_pose_world.orientation.y,
                                                       self.plate_pose_world.orientation.z, self.plate_pose_world.orientation.w]))
                if not raw_orient_valid:
                     rospy.logerr("Non-finite value found in raw plate orientation! Skipping state update.")
                     return False
                raw_twist_valid = np.all(np.isfinite([ball_twist_world.linear.x, ball_twist_world.linear.y, ball_twist_world.linear.z]))
                if not raw_twist_valid:
                     rospy.logerr(f"Non-finite value found in raw ball twist linear: {ball_twist_world.linear}! Using zero velocity.")
                     ball_twist_world.linear.x = 0.0; ball_twist_world.linear.y = 0.0; ball_twist_world.linear.z = 0.0

                # Calculate relative position
                relative_pos_xyz = calculate_relative_position(ball_pose_world.position, self.plate_pose_world)
                relative_pos = relative_pos_xyz[:2]

                # Calculate relative velocity
                plate_orient_q_np = np.array([self.plate_pose_world.orientation.x, self.plate_pose_world.orientation.y,
                                              self.plate_pose_world.orientation.z, self.plate_pose_world.orientation.w])
                if np.sum(np.square(plate_orient_q_np)) < 1e-6: plate_orient_q_np = np.array([0.0, 0.0, 0.0, 1.0])
                else: plate_orient_q_np = plate_orient_q_np / np.linalg.norm(plate_orient_q_np)

                try:
                    plate_orient_q_inv = tf_trans.quaternion_inverse(plate_orient_q_np)
                    ball_vel_world_np = np.array([ball_twist_world.linear.x, ball_twist_world.linear.y, ball_twist_world.linear.z])
                    ball_vel_quat = np.append([0], ball_vel_world_np)
                    ball_vel_plate_frame_quat = tf_trans.quaternion_multiply(
                        tf_trans.quaternion_multiply(plate_orient_q_inv, ball_vel_quat), plate_orient_q_np
                    )
                    relative_vel = ball_vel_plate_frame_quat[1:3]
                except (np.linalg.LinAlgError, ValueError) as e:
                     rospy.logwarn_throttle(5.0, f"Quaternion math error for velocity: {e}.")
                     relative_vel = np.array([0.0, 0.0])

                # Final checks before concatenation
                if not np.all(np.isfinite(relative_pos)):
                    rospy.logerr(f"Non-finite value calculated for relative_pos: {relative_pos}. Using Zeros.")
                    relative_pos = np.array([0.0, 0.0])
                if not np.all(np.isfinite(relative_vel)):
                    rospy.logerr(f"Non-finite value calculated for relative_vel: {relative_vel}. Using Zeros.")
                    relative_vel = np.array([0.0, 0.0])

                # Update observation state and clip
                self.current_observation = np.concatenate([relative_pos, relative_vel]).astype(np.float32)
                self.current_observation = np.clip(
                    self.current_observation, self.observation_space.low, self.observation_space.high
                )
                return True
            else:
                missing = []
                if ball_link_name not in msg.name: missing.append(ball_link_name)
                if plate_link_name not in msg.name: missing.append(plate_link_name)
                if 'table_1::base' not in msg.name: missing.append('table_1::base')
                rospy.logwarn_throttle(5.0, f"Links not found in /gazebo/link_states: {missing}")
                return False

        except Exception as e:
            rospy.logerr(f"Error in _process_latest_link_states: {e}")
            traceback.print_exc()
            return False


    def step(self, action):
        """ Applies action, steps simulation, gets observation, calculates reward. """
        terminated = False
        truncated = False
        info = {}



        # 1. Apply action (joint efforts)
        effort_req = ApplyJointEffortRequest()
        effort_req.start_time = rospy.Time.now()
        effort_req.duration = rospy.Duration(-1) # Continuous until next step command
        effort_x = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        effort_y = np.clip(action[1], self.action_space.low[1], self.action_space.high[1])

        # --- LOG APPLIED EFFORT (Throttled) ---
        rospy.loginfo_throttle(5.0, f"Applying efforts: x={effort_x:.3f}, y={effort_y:.3f}")
        # ------------------------------------

        try:
             effort_req.joint_name = 'joint_x'; effort_req.effort = effort_x
             self.apply_effort_proxy(effort_req)
             effort_req.joint_name = 'joint_y'; effort_req.effort = effort_y
             self.apply_effort_proxy(effort_req)
        except rospy.ServiceException as e:
             rospy.logerr(f"Apply Joint Effort service call failed: {e}")
             terminated = True 
             # If apply effort fails, maybe terminate or return error?
             reward = self.TERMINATION_PENALTY
             observation = np.clip(self.current_observation, self.observation_space.low, self.observation_space.high)
             return observation, reward, terminated, truncated, {"error": "apply_effort_failed"}
             # For now, we proceed, potentially with stale physics

        # 2. Step Simulation & Get State
        time.sleep(0.1) # Use user's increased sleep time


        success = self._process_latest_link_states()
        if not success:
            rospy.logwarn("Failed to process link states in step, using previous observation.")

        observation = np.clip(
            self.current_observation, self.observation_space.low, self.observation_space.high
        )

        rel_x = observation[0]
        rel_y = observation[1]
        # rel_vx = observation[2] # Uncomment if using velocity in reward/termination
        # rel_vy = observation[3] # Uncomment if using velocity in reward/termination

        # 3. Calculate Reward (NEW Centering Reward Logic)
        distance_from_center_sq = rel_x**2 + rel_y**2
        # Exponential reward: Max reward at center, falls off quickly
        centering_reward = self.MAX_CENTERING_REWARD * np.exp(-self.CENTERING_REWARD_SCALE * distance_from_center_sq)

        action_penalty = -self.ACTION_PENALTY_WEIGHT * np.sum(np.square(action)) # Penalty for large actions

        # Combine reward components
        reward = centering_reward + action_penalty + 0.1

        # 4. Check Termination
        ball_z = -1.0
        if self.latest_link_states and self.plate_pose_world:
             try:
                 ball_index = self.latest_link_states.name.index('ball_1::ball_link')
                 ball_z = self.latest_link_states.pose[ball_index].position.z
             except ValueError: 
                rospy.logwarn_throttle(5.0, "Could not find 'ball_1::ball_link' in link_states for termination check.")
                # If ball state is lost, consider it terminated
                terminated = True
                reward = self.TERMINATION_PENALTY # Assign penalty directly

        if ball_z < self.Z_THRESHOLD or abs(rel_x) > self.BOUND_LIMIT or abs(rel_y) > self.BOUND_LIMIT:
            terminated = True
            reward = self.TERMINATION_PENALTY
            rospy.logwarn_throttle(1.0, f"Episode terminated. Reason: Ball off table (Z={ball_z:.3f}, RelX={rel_x:.3f}, RelY={rel_y:.3f})")

        # 5. Check Truncation
        self.step_count += 1
        if self.step_count >= self.MAX_STEPS_PER_EPISODE:
            truncated = True
            rospy.loginfo_throttle(10.0, f"Episode truncated after {self.MAX_STEPS_PER_EPISODE} steps.")
        
        if terminated or truncated: self.step_count = 0

        # 6. Return Values
        return observation, reward, terminated, truncated, info

        print(f"Step {self.step_count}: Agent action: {action}, Clipped efforts: x={effort_x:.3f}, y={effort_y:.3f}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pos_bound = 0.17

        # --- FULL SIMULATION RESET ---
        try:
            rospy.loginfo_throttle(10.0,"Calling full simulation reset service...")
            self.reset_simulation_proxy()
            rospy.loginfo_throttle(10.0,"Full simulation reset completed.")
            # Short pause after full reset might be needed for services to re-register
             
        except rospy.ServiceException as e:
            rospy.logerr(f"Reset: Full simulation reset service call failed: {e}")
        time.sleep(0.5)  

        try:
            rospy.wait_for_service('/gazebo/apply_joint_effort', timeout=2.0)
            rospy.wait_for_service('/gazebo/set_model_state', timeout=2.0)
        except rospy.ROSException as e:
            rospy.logerr(f"Reset: /gazebo/apply_joint_effort service not available: {e}")

        # Generate random x, y for ball spawn
        random_x = self.np_random.uniform(low=-pos_bound, high=pos_bound)
        random_y = self.np_random.uniform(low=-pos_bound, high=pos_bound)

        # Define table state: force it to a level pose
        table_state = ModelState(
            model_name='table_1',
            pose=Pose(
                position=Point(0, 0, 0),
                orientation=Quaternion(0, 0, 0, 1)  # level orientation
            ),
            twist=Twist()
        )
        initial_twist = Twist()
        initial_twist.linear.x = self.np_random.uniform(low=-0.1, high=0.1)
        initial_twist.linear.y = self.np_random.uniform(low=-0.1, high=0.1)

        # Define ball state: drop from a specified height (make this configurable)
        ball_state = ModelState(
            model_name='ball_1',
            pose=Pose(
                position=Point(random_x, random_y, 0.6),  # e.g., self.ball_drop_height = 0.6
                orientation=Quaternion(0, 0, 0,1)
            ),
            twist=initial_twist
        )

        # Pause physics to perform the reset cleanly
        try:
            self.pause_physics_proxy()
        except rospy.ServiceException as e:
            rospy.logerr(f"Reset: Failed to pause physics: {e}")

        # Reset model states for table and ball
        try:
            self.set_model_state_proxy(SetModelStateRequest(model_state=table_state))
            self.set_model_state_proxy(SetModelStateRequest(model_state=ball_state))
        except rospy.ServiceException as e:
            rospy.logerr(f"Reset: Set model state failed: {e}")

        time.sleep(0.1)   

        # Optionally, set zero effort on joints to ensure no residual control commands
        #effort_req = ApplyJointEffortRequest()
        #effort_req.effort = 0.0
        #effort_req.start_time = rospy.Time.now()
        #effort_req.duration = rospy.Duration(0.1)
        #for joint in ['joint_x', 'joint_y']:
        #    effort_req.joint_name = joint
        #    try:
       # #        self.apply_effort_proxy(effort_req)
        #    except rospy.ServiceException as e:
        #        rospy.logerr(f"Reset: Failed zero effort on {joint}: {e}")
#
        ## Unpause physics after resetting
        try:
            self.unpause_physics_proxy()
        except rospy.ServiceException as e:
            rospy.logerr(f"Reset: Failed to unpause physics: {e}")

        time.sleep(0.5)  # Allow time for the simulation to settle

        success = self._process_latest_link_states()
        if not success:
            rospy.logerr("Failed to get initial observation after reset.")
            self.current_observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        observation = np.clip(self.current_observation, self.observation_space.low, self.observation_space.high)
        info = {}
        self.step_count = 0
        return observation, info


    def render(self): pass
    def close(self): rospy.loginfo("Closing TiltingTableEnv"); pass

# --- Main block for testing/checking ---
if __name__ == '__main__':
    env = None
    try:
        rospy.loginfo("Creating environment...")
        env = TiltingTableEnv()
        rospy.loginfo("Environment created successfully!")
        print("Action Space:", env.action_space)
        print("Observation Space:", env.observation_space)

        # Test reset
        print("Testing reset...")
        obs, info = env.reset()
        print("Reset observation:", obs)

        # Test step
        print("Testing step with zero action...")
        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step result: obs={obs}, reward={reward:.3f}, term={terminated}, trunc={truncated}, info={info}")

        try:
            rospy.loginfo("Checking environment compatibility...")
            check_env(env, warn=True, skip_render_check=True)
            rospy.loginfo("Environment check passed!")
        except Exception as e:
            rospy.logerr("Environment check failed!")
            traceback.print_exc()

    except rospy.ROSInterruptException: rospy.loginfo("ROS Interrupt received during testing.")
    except Exception as e: rospy.logerr(f"Error during environment testing: {e}"); traceback.print_exc()
    finally:
        if env is not None: env.close()
        rospy.loginfo("Testing script finished.")