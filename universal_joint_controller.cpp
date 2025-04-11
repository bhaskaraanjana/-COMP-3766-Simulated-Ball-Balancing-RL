#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <thread>

namespace gazebo
{
  class UniversalJointController : public ModelPlugin
  {
    public: UniversalJointController() : ModelPlugin() {}

    public: virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      // Store the model pointer
      this->model = _model;
      
      // Initialize ROS if it hasn't been initialized already
      if (!ros::isInitialized())
      {
         int argc = 0;
         char **argv = NULL;
         ros::init(argc, argv, "universal_joint_controller", ros::init_options::NoSigintHandler);
      }
      
      // Create a ROS node handle
      this->rosNode.reset(new ros::NodeHandle("universal_joint_controller"));

      // Retrieve the joint name from the SDF element
      if (_sdf->HasElement("joint_name"))
      {
         this->jointName = _sdf->Get<std::string>("joint_name");
      }
      else
      {
         gzerr << "[UniversalJointController] Missing <joint_name> in SDF.\n";
         return;
      }

      // Get the joint from the model
      this->joint = this->model->GetJoint(this->jointName);
      if (!this->joint)
      {
         gzerr << "[UniversalJointController] Joint '" << this->jointName << "' not found.\n";
         return;
      }

      // Subscribe to the tilt command topic.
      // The expected message is a Float64MultiArray with 2 elements:
      // - data[0] corresponds to joint_x effort (axis1)
      // - data[1] corresponds to joint_y effort (axis2)
      std::string topicName = "/tilting_table/tilt_cmd";
      this->rosSub = this->rosNode->subscribe(topicName, 1,
                      &UniversalJointController::OnRosMsg, this);

      // Connect to the world update event (called every simulation iteration)
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&UniversalJointController::OnUpdate, this));

      gzdbg << "[UniversalJointController] Plugin loaded for joint: " << this->jointName << "\n";
    }

    // Called every simulation iteration
    public: void OnUpdate()
    {
      // For a universal joint, axis 0 corresponds to the first axis (joint_x)
      // and axis 1 corresponds to the second axis (joint_y).
      if (this->joint)
      {
         this->joint->SetForce(0, this->joint_x);
         this->joint->SetForce(1, this->joint_y);
      }
    }

    // Callback for receiving ROS commands
    public: void OnRosMsg(const std_msgs::Float64MultiArray::ConstPtr &msg)
    {
       if (msg->data.size() < 2)
       {
          gzerr << "[UniversalJointController] Received insufficient data. Expected 2 values, got " << msg->data.size() << "\n";
          return;
       }
       // Set our control variables using the incoming data.
       this->joint_x = msg->data[0];
       this->joint_y = msg->data[1];
       gzdbg << "[UniversalJointController] Received command: joint_x = " << this->joint_x
             << ", joint_y = " << this->joint_y << "\n";
    }

    private: physics::ModelPtr model;
    private: physics::JointPtr joint;
    private: std::string jointName;
    private: event::ConnectionPtr updateConnection;
    private: std::unique_ptr<ros::NodeHandle> rosNode;
    private: ros::Subscriber rosSub;

    // Control variables for the two axes of the universal joint.
    private: double joint_x{0.0};
    private: double joint_y{0.0};
  };

  // Register the plugin with Gazebo
  GZ_REGISTER_MODEL_PLUGIN(UniversalJointController)
}
