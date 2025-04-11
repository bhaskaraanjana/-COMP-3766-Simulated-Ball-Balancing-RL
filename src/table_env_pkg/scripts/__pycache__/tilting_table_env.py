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
    # Handle potential zero quaternion if plate state is invalid
    if np.all(plate_orient_q == 0):
        plate_orient_q = np.array([0.0, 0.0, 0.0, 1.0]) # Default to identity
        rospy.logwarn_throttle(5.0,"Plate orientation quaternion is zero, using identity.")

    try:
        plate_orient_q_inv = tf_trans.quaternion_inverse(plate_orient_q)
        diff_quat = np.append([0], diff_world)
        diff_plate_frame_quat = tf_trans.quaternion_multiply(
            tf_trans.quaternion_multiply(plate_orient_q_inv, diff_quat),
            plate_orient_q
        )
        relative_pos = diff_plate_frame_quat[1:] # Get XYZ relative position
    except np.linalg.LinAlgError: # Catch potential errors in quaternion inverse if norm is zero
         rospy.logwarn_throttle(5.0,"Singular quaternion encountered for plate orientation, using world difference.")
         relative_pos = diff_world # Fallback if rotation fails

    return relative_pos # Returns numpy array [x, y, z]

# --- Environment Class ---
class TiltingTableEnv(gym.Env):
    metadata = {'render_modes': ['human'], "render_fps": 30}
    SURVIVAL_BONUS = 0.01
    TERMINATION_PENALTY = -10.0
    BOUND_LIMIT = 0.21
    Z_THRESHOLD = 0.08 # Use lower threshold
    MAX_STEPS_PER_EPISODE = 1000
    ACTION_PENALTY_WEIGHT = 0.001 # Added action penalty weight

    def __init__(self):
        super(TiltingTableEnv, self).__init__()

        max_effort = 0.1 # Keep reduced max effort
        self.action_space = spaces.Box(low=-max_effort, high=max_effort, shape=(2,), dtype=np.float32)
        obs_limit = np.array([0.2, 0.2, 5.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_limit, high=obs_limit, shape=(4,), dtype=np.float32)

        try:
            if not rospy.core.is_initialized():
                rospy.init_node('tilting_table_env', anonymous=True)
                self.node_initialized_by_env = True
            else:
                self.node_initialized_by_env = False
        except rospy.exceptions.ROSException as e:
             rospy.logwarn(f"ROS node already initialized: {e}")
             self.node_initialized_by_env = False

        rospy.loginfo("Waiting for Gazebo services...")
        rospy.wait_for_service('/gazebo/apply_joint_effort')
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/pause_physics')
        rospy.wait_for_service('/gazebo/unpause_physics')
        self.apply_effort_proxy = rospy.ServiceProxy('/gazebo/apply_joint_effort', ApplyJointEffort)
        self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        rospy.loginfo("Gazebo services connected.")

        self.latest_link_states = None
        rospy.Subscriber('/gazebo/link_states', LinkStates, self._link_states_callback)
        rospy.loginfo("Subscribed to /gazebo/link_states")

        self.current_observation = np.zeros(4, dtype=np.float32)
        self.plate_pose_world = None
        self.rate = rospy.Rate(100)
        self.step_count = 0

        rospy.loginfo("Waiting for initial link state message...")
        start_wait_time = rospy.get_time()
        while self.latest_link_states is None and not rospy.is_shutdown():
             if rospy.get_time() - start_wait_time > 5.0:
                 rospy.logerr("Timeout waiting for initial /gazebo/link_states message.")
                 sys.exit()
             try: self.rate.sleep()
             except rospy.exceptions.ROSInterruptException: rospy.loginfo("Shutdown requested."); return
        rospy.loginfo("Initial link state received.")
        self._process_latest_link_states()


    def _link_states_callback(self, msg):
        """ Stores the latest LinkStates message. """
        self.latest_link_states = msg


    def _process_latest_link_states(self):
        """ Processes the stored LinkStates msg to update observation, with detailed checks. """
        if self.latest_link_states is None:
            rospy.logwarn_throttle(5.0,"Cannot process link states, none received yet.")
            # Return False, but ensure observation is zero if called before first valid state
            self.current_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            return False

        msg = self.latest_link_states
        try:
            ball_link_name = 'ball_1::ball_link'
            plate_link_name = 'table_1::plate'

            if ball_link_name in msg.name and plate_link_name in msg.name:
                ball_index = msg.name.index(ball_link_name)
                plate_index = msg.name.index(plate_link_name)

                # --- Check Raw Data Validity ---
                ball_pose_world = msg.pose[ball_index]
                ball_twist_world = msg.twist[ball_index]
                self.plate_pose_world = msg.pose[plate_index]

                # Check positions (unlikely to be inf/nan, but good practice)
                raw_poses_valid = np.isfinite(ball_pose_world.position.x) and \
                                  np.isfinite(ball_pose_world.position.y) and \
                                  np.isfinite(ball_pose_world.position.z) and \
                                  np.isfinite(self.plate_pose_world.position.x) and \
                                  np.isfinite(self.plate_pose_world.position.y) and \
                                  np.isfinite(self.plate_pose_world.position.z)
                if not raw_poses_valid:
                     rospy.logerr("Non-finite value found in raw link positions! Skipping state update.")
                     # Keep previous observation? Or return zeros? Let's keep previous for now.
                     return False # Indicate failure to update

                # Check orientation (unlikely unless simulation exploded)
                raw_orient_valid = np.all(np.isfinite([self.plate_pose_world.orientation.x, self.plate_pose_world.orientation.y,
                                                       self.plate_pose_world.orientation.z, self.plate_pose_world.orientation.w]))
                if not raw_orient_valid:
                     rospy.logerr("Non-finite value found in raw plate orientation! Skipping state update.")
                     return False

                # Check twist (most likely source of inf/nan if physics unstable)
                raw_twist_valid = np.isfinite(ball_twist_world.linear.x) and \
                                  np.isfinite(ball_twist_world.linear.y) and \
                                  np.isfinite(ball_twist_world.linear.z)
                if not raw_twist_valid:
                     rospy.logerr(f"Non-finite value found in raw ball twist linear: x={ball_twist_world.linear.x}, y={ball_twist_world.linear.y}, z={ball_twist_world.linear.z}! Using zero velocity.")
                     # Set twist to zero if invalid to prevent calculation errors
                     ball_twist_world.linear.x = 0.0
                     ball_twist_world.linear.y = 0.0
                     ball_twist_world.linear.z = 0.0
                # --- End Raw Data Check ---


                # Calculate relative position
                relative_pos_xyz = calculate_relative_position(ball_pose_world.position, self.plate_pose_world)
                relative_pos = relative_pos_xyz[:2] # Get only X and Y

                # Calculate relative velocity
                plate_orient_q_np = np.array([self.plate_pose_world.orientation.x, self.plate_pose_world.orientation.y,
                                              self.plate_pose_world.orientation.z, self.plate_pose_world.orientation.w])
                if np.all(plate_orient_q_np == 0): plate_orient_q_np = np.array([0.0, 0.0, 0.0, 1.0])

                try:
                    plate_orient_q_inv = tf_trans.quaternion_inverse(plate_orient_q_np)
                    ball_vel_world_np = np.array([ball_twist_world.linear.x, ball_twist_world.linear.y, ball_twist_world.linear.z])
                    ball_vel_quat = np.append([0], ball_vel_world_np)
                    ball_vel_plate_frame_quat = tf_trans.quaternion_multiply(
                        tf_trans.quaternion_multiply(plate_orient_q_inv, ball_vel_quat), plate_orient_q_np
                    )
                    relative_vel = ball_vel_plate_frame_quat[1:3] # Get only X and Y relative velocity
                except np.linalg.LinAlgError:
                     rospy.logwarn_throttle(5.0, "Singular quaternion encountered for velocity calculation.")
                     relative_vel = np.array([0.0, 0.0]) # Fallback

                # Check calculated values before concatenation
                if not np.all(np.isfinite(relative_pos)):
                    rospy.logerr(f"Non-finite value calculated for relative_pos: {relative_pos}")
                    relative_pos = np.nan_to_num(relative_pos, nan=0.0, posinf=0.0, neginf=0.0)
                if not np.all(np.isfinite(relative_vel)):
                    rospy.logerr(f"Non-finite value calculated for relative_vel: {relative_vel}")
                    relative_vel = np.nan_to_num(relative_vel, nan=0.0, posinf=0.0, neginf=0.0)

                # Update observation state
                self.current_observation = np.concatenate([relative_pos, relative_vel]).astype(np.float32)
                self.current_observation = np.clip(
                    self.current_observation, self.observation_space.low, self.observation_space.high
                )
                # rospy.logwarn(f"[DEBUG process] Calculated Observation (Clipped): {self.current_observation}") # Keep commented unless needed
                return True
            else:
                missing = []
                if ball_link_name not in msg.name: missing.append(ball_link_name)
                if plate_link_name not in msg.name: missing.append(plate_link_name)
                if 'table_1::base' not in msg.name: missing.append('table_1::base') # Check base just in case
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
        effort_req.start_time = rospy.Time(0)
        effort_req.duration = rospy.Duration(-1) # Continuous until next step command
        effort_x = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        effort_y = np.clip(action[1], self.action_space.low[1], self.action_space.high[1])

        # --- LOG APPLIED EFFORT ---
        rospy.loginfo_throttle(5.0, f"Applying efforts: x={effort_x:.3f}, y={effort_y:.3f}") # Log effort every 5s
        # ---------------------------

        effort_req.joint_name = 'joint_x'; effort_req.effort = effort_x
        try: self.apply_effort_proxy(effort_req)
        except rospy.ServiceException as e: rospy.logerr(f"Failed to apply effort to joint_x: {e}")

        effort_req.joint_name = 'joint_y'; effort_req.effort = effort_y
        try: self.apply_effort_proxy(effort_req)
        except rospy.ServiceException as e: rospy.logerr(f"Failed to apply effort to joint_y: {e}")

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

        # 3. Calculate Reward
        # Inside step method, reward calculation:
        distance_penalty = - (rel_x**2 + rel_y**2)
        # Increase this weight significantly:
        action_penalty_weight = 0.1 # Try 0.01, 0.05, or 0.1
        action_penalty = -action_penalty_weight * np.sum(np.square(action))
        reward = distance_penalty + self.SURVIVAL_BONUS + action_penalty

        # 4. Check Termination
        ball_z = -1.0
        if self.latest_link_states:
             try:
                 ball_index = self.latest_link_states.name.index('ball_1::ball_link')
                 ball_z = self.latest_link_states.pose[ball_index].position.z
             except ValueError: pass

        if ball_z < self.Z_THRESHOLD or abs(rel_x) > self.BOUND_LIMIT or abs(rel_y) > self.BOUND_LIMIT:
            terminated = True
            reward += self.TERMINATION_PENALTY
            # Termination reason logging (throttled)
            rospy.logwarn_throttle(1.0, f"Episode terminated. Reason: Ball off table (Z={ball_z:.3f}, RelX={rel_x:.3f}, RelY={rel_y:.3f})")

        # 5. Check Truncation
        self.step_count += 1
        if self.step_count >= self.MAX_STEPS_PER_EPISODE:
            truncated = True
        if terminated or truncated: self.step_count = 0

        # 6. Return Values
        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        """ Resets the environment, lets ball settle, returns initial observation. """
        super().reset(seed=seed)
        rospy.logdebug("Resetting environment...")
        pos_bound = 0.17

        # Reset model states
        random_x = self.np_random.uniform(low=-pos_bound, high=pos_bound)
        random_y = self.np_random.uniform(low=-pos_bound, high=pos_bound)
        ball_state = ModelState(model_name='ball_1', pose=Pose(position=Point(random_x, random_y, 0.13), orientation=Quaternion(0,0,0,1)), twist=Twist())
        table_state = ModelState(model_name='table_1', pose=Pose(position=Point(0,0,0), orientation=Quaternion(0,0,0,1)), twist=Twist())
        try:
            self.set_model_state_proxy(SetModelStateRequest(model_state=table_state))
            self.set_model_state_proxy(SetModelStateRequest(model_state=ball_state))
        except rospy.ServiceException as e: rospy.logerr(f"Reset: Set model state failed: {e}")

        # Apply Zero Effort Explicitly & Wait for Settle
        effort_req = ApplyJointEffortRequest()
        effort_req.joint_name = 'joint_x'; effort_req.effort = 0.0
        effort_req.start_time = rospy.Time.now()
        effort_req.duration = rospy.Duration(0.1) # Apply zero for 0.1s
        try: self.apply_effort_proxy(effort_req)
        except rospy.ServiceException as e: rospy.logerr(f"Reset: Failed zero effort joint_x: {e}")
        effort_req.joint_name = 'joint_y'
        try: self.apply_effort_proxy(effort_req)
        except rospy.ServiceException as e: rospy.logerr(f"Reset: Failed zero effort joint_y: {e}")

        time.sleep(0.3) # Use user's increased settle time

        # Get Initial Observation
        success = self._process_latest_link_states()
        if not success:
             rospy.logerr("Failed to get initial observation after reset.")
             self.current_observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        observation = np.clip(
            self.current_observation, self.observation_space.low, self.observation_space.high
        )
        info = {}
        self.step_count = 0

        # rospy.logdebug(f"Environment Reset. Initial Obs: {observation}")
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
