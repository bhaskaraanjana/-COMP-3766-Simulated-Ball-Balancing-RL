#!/usr/bin/env python3
import rospy
import os
import datetime # Added for timing
from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env # Keep commented unless using >1 env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from tilting_table_env import TiltingTableEnv # Make sure this uses the version with reset fixes

# --- Parameters ---
LOG_DIR = os.path.expanduser("~/catkin_ws/training_logs")
MODEL_SAVE_NAME = "ppo_tilting_table" # Base name for model files
FINAL_MODEL_PATH = os.path.join(LOG_DIR, MODEL_SAVE_NAME + "_final.zip") # Full path for final model
TOTAL_TIMESTEPS = 1000000 # Increased total timesteps (adjust as needed)

def main():
    # Create log directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)

    # Initialize ROS node only once
    # Note: TiltingTableEnv also tries to init node, ensure this doesn't conflict
    # or remove the init from __init__ in TiltingTableEnv if running this script always
    if not rospy.core.is_initialized():
        rospy.init_node('ppo_trainer', anonymous=True)
        rospy.loginfo("PPO Trainer Node Initialized.")
    else:
        rospy.loginfo("PPO Trainer using existing ROS node.")

    # Record training start time (wall-clock time)
    start_time = datetime.datetime.now()
    print(f"Training started at: {start_time}") # Use standard print for wall time

    # --- Create the environments ---
    # Ensure TiltingTableEnv uses the corrected observation space if loading old model,
    # or the desired space if training from scratch.
    try:
        env = TiltingTableEnv()
        eval_env = TiltingTableEnv()
    except Exception as e:
        rospy.logerr(f"Failed to create TiltingTableEnv instances: {e}")
        rospy.logerr("Ensure ROS Master and Gazebo are running properly.")
        return # Exit if environment creation fails

    # --- Callbacks ---
    # Evaluation Callback: Periodically evaluates policy on eval_env, saves best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=LOG_DIR, # Saves best model to LOG_DIR/best_model.zip
        log_path=LOG_DIR,             # Saves evaluation results to LOG_DIR/evaluations.npz
        eval_freq=10000,              # Evaluate every 10k timesteps (adjust as needed)
        n_eval_episodes=5,            # Run 5 evaluation episodes
        deterministic=True,
        render=False
    )

    # Checkpoint Callback: Saves model periodically during training
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,              # Save a checkpoint every 20k timesteps (adjust as needed)
        save_path=LOG_DIR,
        name_prefix=MODEL_SAVE_NAME    # Checkpoint files named like ppo_tilting_table_10000_steps.zip
        )

    # --- Device Selection ---
    device_to_use = "cpu" # Explicitly set to CPU
    rospy.loginfo(f"Using device: {device_to_use}")

    # --- Create or Load the PPO Agent ---
    try:
        if os.path.exists(FINAL_MODEL_PATH):
            rospy.loginfo("Loading checkpoint from %s", FINAL_MODEL_PATH)
            # Pass the device argument when loading
            model = PPO.load(FINAL_MODEL_PATH, env=env, device=device_to_use, tensorboard_log=LOG_DIR)
            # Reset TensorBoard Timesteps (optional, if you want TB graphs to restart)
            # model.set_log_dir(LOG_DIR) # Creates a new PPO_x folder if needed
        else:
            rospy.loginfo("No checkpoint found, starting training from scratch.")
            # Pass the device argument when creating a new model
            model = PPO("MlpPolicy",
                        env,
                        verbose=1,
                        tensorboard_log=LOG_DIR,
                        learning_rate=0.0001, # Example LR, adjust as needed (e.g., 3e-4)
                        n_steps=512,         # Example, adjust based on episode length/performance
                        batch_size=64,        # Example, often 64
                        # gamma=0.99,         # Default usually okay
                        # vf_coef=0.5,        # Default, adjust if needed
                        # policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]), # Example larger network
                        device=device_to_use)

    except Exception as e:
        rospy.logerr(f"Failed to create or load PPO model: {e}")
        traceback.print_exc() # Print detailed error
        env.close()
        eval_env.close()
        return # Exit if model creation/loading fails


    # --- Train the agent ---
    rospy.loginfo(f"Starting PPO training for {TOTAL_TIMESTEPS} timesteps...")
    combined_callbacks = [checkpoint_callback, eval_callback] # Use both callbacks

    try:
        # The learn method starts the training loop
        # Pass the list of callbacks here
        model.learn(total_timesteps=TOTAL_TIMESTEPS,
                    callback=combined_callbacks,
                    log_interval=1) # Log training info every N updates (1 = often)

        rospy.loginfo("Training finished!")
        # Record and print elapsed time
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print(f"Training completed in: {elapsed_time}") # Use standard print for wall time

        # --- Save the final trained model ---
        model.save(FINAL_MODEL_PATH) # SB3 automatically adds .zip
        rospy.loginfo(f"Final model saved to {FINAL_MODEL_PATH}")

    except rospy.ROSInterruptException:
        rospy.logwarn("Training interrupted by ROS shutdown.")
    except KeyboardInterrupt:
        rospy.logwarn("Training interrupted by user (Ctrl+C). Saving current model...")
        # Save model even if interrupted by Ctrl+C
        model.save(FINAL_MODEL_PATH)
        rospy.loginfo(f"Interim model saved to {FINAL_MODEL_PATH}")
    except Exception as e:
        rospy.logerr(f"An error occurred during training: {e}")
        traceback.print_exc()
        # Optionally try saving model on other errors too
        # model.save(FINAL_MODEL_PATH)
        # rospy.loginfo(f"Attempted to save model after error to {FINAL_MODEL_PATH}")
    finally:
        # Close the environments cleanly
        env.close()
        eval_env.close() # Close eval env too
        rospy.loginfo("Environments closed.")

if __name__ == '__main__':
    main()