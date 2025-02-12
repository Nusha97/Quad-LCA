#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>

#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/CameraInfo.h>


#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>

#include <boost/filesystem.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include<kr_mav_msgs/PositionCommand.h>
#include<kr_mav_msgs/SO3Command.h>
#include<nav_msgs/Odometry.h>
#include<nav_msgs/Odometry.h>

typedef message_filters::sync_policies::ApproximateTime<kr_mav_msgs::PositionCommand, kr_mav_msgs::SO3Command, nav_msgs::Odometry> SyncPolicySim;
typedef message_filters::sync_policies::ApproximateTime<kr_mav_msgs::PositionCommand, kr_mav_msgs::SO3Command, nav_msgs::Odometry, nav_msgs::Odometry> SyncPolicyHw;

class SyncMsg
{
public:
  SyncMsg();
  ~SyncMsg();

private:
  void callback_sim(const kr_mav_msgs::PositionCommand::ConstPtr& pos_cmd, const kr_mav_msgs::SO3Command::ConstPtr& so3_cmd, const nav_msgs::Odometry::ConstPtr& odom);
  void callback_hw(const kr_mav_msgs::PositionCommand::ConstPtr& pos_cmd, const kr_mav_msgs::SO3Command::ConstPtr& so3_cmd, const nav_msgs::Odometry::ConstPtr& odom, const nav_msgs::Odometry::ConstPtr& vicon_odom);

  ros::NodeHandle nh_, pnh_;

  message_filters::Subscriber<kr_mav_msgs::PositionCommand> sub_t1_;
  message_filters::Subscriber<kr_mav_msgs::SO3Command> sub_t2_;
  message_filters::Subscriber<nav_msgs::Odometry> sub_t3_;
  message_filters::Subscriber<nav_msgs::Odometry> sub_t4_;

  std::shared_ptr< message_filters::Synchronizer<SyncPolicySim> > sync_sim_;
  std::shared_ptr< message_filters::Synchronizer<SyncPolicyHw> > sync_hw_;

  std::shared_ptr<rosbag::Bag> write_bag_pt_;

  std::string namespace_, bag_file_;
  std:: string t1_, t2_, t3_, t4_;

  int count_;
  bool sim_;

};

SyncMsg::SyncMsg(): pnh_("~"), count_(1)
{

  pnh_.param<std::string>("namespace", namespace_, "dragonfly1");
  pnh_.param<bool>("sim", sim_, true);

  time_t curr_time;
  tm * curr_tm;
  char date_string[100];
  char time_string[100];

  time(&curr_time);
  curr_tm = localtime(&curr_time);

  strftime(date_string, 50, "%F", curr_tm);
  strftime(time_string, 50, "%H-%M-%S", curr_tm);

  std::string bag_name = namespace_ + "-" + std::string(date_string) + "-" + std::string(time_string) + ".bag";
  std::string home_dir = getenv("HOME");
  int counter = 0;

  bag_file_ = home_dir + "/" + bag_name;

  ROS_INFO("Namespace %s", namespace_.c_str());
  ROS_INFO("Bag storage location %s", home_dir.c_str());
  ROS_INFO("Writing to bag %s", bag_file_.c_str());

  write_bag_pt_.reset(new rosbag::Bag());
  write_bag_pt_->open(bag_file_, rosbag::bagmode::Write);

  t1_ = "/" + namespace_ + "/position_cmd";
  t2_ = "/" + namespace_ + "/so3_cmd";
  t3_ = "/" + namespace_ + "/odom";

  if(!sim_)
  {
    t3_ = "/" + namespace_ + "/quadrotor_ukf/control_odom";
    t4_ = "/vicon/" + namespace_ + "/odom";
  }

  sub_t1_.subscribe(nh_, t1_, 10);
  sub_t2_.subscribe(nh_, t2_, 10);
  sub_t3_.subscribe(nh_, t3_, 10);
  sub_t4_.subscribe(nh_, t4_, 10);

  if(sim_)
  {
    sync_sim_.reset(new message_filters::Synchronizer<SyncPolicySim> (SyncPolicySim(20), sub_t1_, sub_t2_, sub_t3_));
    sync_sim_->registerCallback(boost::bind(&SyncMsg::callback_sim, this, _1, _2, _3));

    ROS_INFO("Sim: Subscribing and syncing topics %s %s %s", t1_.c_str(), t2_.c_str(), t3_.c_str());
  }
  else
  {
    sync_hw_.reset(new message_filters::Synchronizer<SyncPolicyHw> (SyncPolicyHw(20), sub_t1_, sub_t2_, sub_t3_, sub_t4_));
    sync_hw_->registerCallback(boost::bind(&SyncMsg::callback_hw, this, _1, _2, _3, _4));

    ROS_INFO("Hw: Subscribing and syncing topics %s %s %s %s", t1_.c_str(), t2_.c_str(), t3_.c_str(), t4_.c_str());
  }

}

SyncMsg::~SyncMsg()
{
  ROS_WARN("%d messages synced to bag %s",count_, bag_file_.c_str());
  write_bag_pt_->close();
}

void SyncMsg::callback_sim(const kr_mav_msgs::PositionCommand::ConstPtr& pos_cmd, const kr_mav_msgs::SO3Command::ConstPtr& so3_cmd, const nav_msgs::Odometry::ConstPtr& odom)
{
  ROS_WARN("sync callback %d", count_);

  write_bag_pt_->write(t1_,pos_cmd->header.stamp, pos_cmd);
  write_bag_pt_->write(t2_,so3_cmd->header.stamp, so3_cmd);
  write_bag_pt_->write(t3_,odom->header.stamp, odom);
  count_++;
}

void SyncMsg::callback_hw(const kr_mav_msgs::PositionCommand::ConstPtr& pos_cmd, const kr_mav_msgs::SO3Command::ConstPtr& so3_cmd, const nav_msgs::Odometry::ConstPtr& odom, const nav_msgs::Odometry::ConstPtr& vicon_odom)
{
  ROS_WARN("sync callback %d", count_);

  write_bag_pt_->write(t1_,pos_cmd->header.stamp, pos_cmd);
  write_bag_pt_->write(t2_,so3_cmd->header.stamp, so3_cmd);
  write_bag_pt_->write(t3_,odom->header.stamp, odom);
  write_bag_pt_->write(t4_,vicon_odom->header.stamp, vicon_odom);
  count_++;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "sync_node");
  ros::NodeHandle nh;

  SyncMsg syn_msg;
  ROS_WARN("init done");
    
  ros::spin();

  return 0;
}
