#!/usr/bin/env python3
import rospy
import os
import numpy as np
from stable_baselines3 import PPO
from tilting_table_env import TiltingTableEnv # Import your environment class
import time

# --- Parameters ---
# IMPORTANT: Specify the exact path to the model file you want to test
MODEL_PATH = os.path.expanduser("~/catkin_ws/trained_models/ppo_tilting_table.zip")
# Or, if testing a checkpoint:
# MODEL_PATH = os.path.expanduser("~/catkin_ws/training_logs/ppo_tilting_table_100000_steps.zip")

# Number of episodes or steps to test for
N_TEST_EPISODES = 10

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()

    rospy.init_node('ppo_tester', anonymous=True)
    rospy.loginfo("PPO Tester Node Started")

    # --- Create the environment ---
    # !!! CRITICAL !!!
    # Ensure this TiltingTableEnv() instance has the SAME observation space definition
    # as the one used when the model at MODEL_PATH was trained.
    # If MODEL_PATH used the 8D observation space, the current env code is likely correct.
    # If MODEL_PATH used the older 4D space, you need to temporarily modify
    # TiltingTableEnv's __init__ to define the 4D observation space before loading.
    try:
        env = TiltingTableEnv()
        rospy.loginfo(f"Using Observation Space: {env.observation_space}")
    except Exception as e:
        rospy.logerr(f"Failed to create TiltingTableEnv: {e}")
        exit()

    # --- Load the trained agent ---
    try:
        # Load the model, explicitly setting the device
        model = PPO.load(MODEL_PATH, env=env, device='cpu')
        rospy.loginfo(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        rospy.logerr(f"Failed to load model from {MODEL_PATH}: {e}")
        rospy.logerr("Ensure the environment's observation/action spaces match the saved model!")
        env.close()
        exit()

    # --- Run testing loop ---
    rospy.loginfo(f"Starting testing for {N_TEST_EPISODES} episodes...")
    total_reward_sum = 0
    total_steps_sum = 0

    for episode in range(N_TEST_EPISODES):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        episode_steps = 0

        while not terminated and not truncated:
            # Get action from the loaded policy (deterministic=True means no exploration noise)
            action, _states = model.predict(obs, deterministic=True)

            # Apply action to the environment
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_steps += 1

            # Optional: Add a small sleep to slow down execution for visualization
            # time.sleep(0.05)

            # Check for ROS shutdown request
            if rospy.is_shutdown():
                rospy.logwarn("ROS shutdown requested during testing.")
                terminated = True # Exit loop

        rospy.loginfo(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")
        total_reward_sum += episode_reward
        total_steps_sum += episode_steps

        if rospy.is_shutdown():
            break # Exit outer loop if ROS shut down

    # --- Clean up ---
    env.close()
    rospy.loginfo("Testing finished.")
    if N_TEST_EPISODES > 0:
      avg_reward = total_reward_sum / N_TEST_EPISODES
      avg_steps = total_steps_sum / N_TEST_EPISODES
      rospy.loginfo(f"Average Reward over {N_TEST_EPISODES} episodes: {avg_reward:.2f}")
      rospy.loginfo(f"Average Steps over {N_TEST_EPISODES} episodes: {avg_steps:.1f}")