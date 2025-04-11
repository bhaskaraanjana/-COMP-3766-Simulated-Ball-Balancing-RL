#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelStates # Message type for model states
from geometry_msgs.msg import Point, Pose
import tf.transformations as tf_trans
import numpy as np

# Global variables to store latest data
ball_pos_world = None
plate_pose_world = None

def calculate_relative_position(ball_world_pos, plate_world_pose):
    """ Calculates ball position relative to the plate frame. """
    plate_pos = np.array([plate_world_pose.position.x, plate_world_pose.position.y, plate_world_pose.position.z])
    plate_orient_q = np.array([plate_world_pose.orientation.x, plate_world_pose.orientation.y,
                            plate_world_pose.orientation.z, plate_world_pose.orientation.w])

    ball_pos = np.array([ball_world_pos.x, ball_world_pos.y, ball_world_pos.z])
    diff_world = ball_pos - plate_pos
    plate_orient_q_inv = tf_trans.quaternion_inverse(plate_orient_q)
    diff_quat = np.append([0], diff_world)
    diff_plate_frame_quat = tf_trans.quaternion_multiply(
        tf_trans.quaternion_multiply(plate_orient_q_inv, diff_quat),
        plate_orient_q
    )
    relative_pos = diff_plate_frame_quat[1:]
    return relative_pos # Returns numpy array [x, y, z]

def model_states_callback(msg):
    """ Callback processing ModelStates messages. """
    global ball_pos_world, plate_pose_world
    try:
        ball_model_name = 'ball_1'
        plate_model_name = 'table_1' # Use the model name from rostopic echo

        if ball_model_name in msg.name and plate_model_name in msg.name:
            ball_index = msg.name.index(ball_model_name)
            plate_index = msg.name.index(plate_model_name)

            ball_pos_world = msg.pose[ball_index].position
            plate_pose_world = msg.pose[plate_index] # Need full pose

            relative_pos_xyz = calculate_relative_position(ball_pos_world, plate_pose_world)

            rospy.loginfo_throttle(1.0, f"Ball Relative Position: x={relative_pos_xyz[0]:.3f}, y={relative_pos_xyz[1]:.3f}")

        else:
            missing = []
            if ball_model_name not in msg.name: missing.append(ball_model_name)
            if plate_model_name not in msg.name: missing.append(plate_model_name)
            rospy.logwarn_throttle(5.0, f"Models not found in /gazebo/model_states: {missing}")

    except Exception as e:
        rospy.logerr(f"Error in model_states_callback: {e}")

def main():
    """ Initializes the node and subscriber. """
    rospy.init_node('ball_tracker_node', anonymous=True)
    rospy.loginfo("Ball Tracker Node Started")
    rospy.Subscriber('/gazebo/model_states', ModelStates, model_states_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Ball Tracker Node Shutdown")
        pass
