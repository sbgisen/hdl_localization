#ifndef GLOBALMAP_SERVER_NODELET_H
#define GLOBALMAP_SERVER_NODELET_H

#include <mutex>
#include <memory>
#include <iostream>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <tf/transform_broadcaster.h>

#include <std_msgs/String.h>
#include <sensor_msgs/PointCloud2.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl/filters/voxel_grid.h>

struct EIGEN_ALIGN16 PointXYZRGBI
{
  PCL_ADD_POINT4D
  PCL_ADD_RGB;
  PCL_ADD_INTENSITY;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZRGBI,
                                  (float, x, x)(float, y, y)(float, z, z)(float, rgb, rgb)(float, intensity, intensity))

namespace hdl_localization
{

class GlobalmapServerNodelet : public nodelet::Nodelet
{
public:
  using PointT = PointXYZRGBI;

  GlobalmapServerNodelet();
  ~GlobalmapServerNodelet() override;
  void onInit() override;

private:
  void initializeParams();
  void pubOnceCb(const ros::WallTimerEvent& event);
  void mapUpdateCallback(const std_msgs::String& msg);

  // ROS
  ros::NodeHandle nh_;
  ros::NodeHandle mt_nh_;
  ros::NodeHandle private_nh_;

  std::string global_frame_id_;
  ros::Publisher globalmap_pub_;
  ros::Subscriber map_update_sub_;

  ros::WallTimer globalmap_pub_timer_;
  pcl::PointCloud<PointT>::Ptr globalmap_;
};

}  // namespace hdl_localization

#endif  // GLOBALMAP_SERVER_NODELET_H
