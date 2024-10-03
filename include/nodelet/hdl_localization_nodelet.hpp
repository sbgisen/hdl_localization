#ifndef HDL_LOCALIZATION_NODELET_H
#define HDL_LOCALIZATION_NODELET_H

#include <mutex>
#include <memory>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <eigen_conversions/eigen_msg.h>

#include <std_srvs/Empty.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <pcl/filters/voxel_grid.h>

#include <pclomp/ndt_omp.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>

#include <hdl_localization/pose_estimator.hpp>
#include <hdl_localization/delta_estimator.hpp>

#include <hdl_localization/ScanMatchingStatus.h>
#include <hdl_global_localization/SetGlobalMap.h>
#include <hdl_global_localization/QueryGlobalLocalization.h>

namespace hdl_localization
{
class HdlLocalizationNodelet : public nodelet::Nodelet
{
public:
  using PointT = pcl::PointXYZI;

  HdlLocalizationNodelet();
  ~HdlLocalizationNodelet() override;

  void onInit() override;

private:
  pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr createRegistration() const;
  void initializeParams();
  void imuCallback(const sensor_msgs::ImuConstPtr& imu_msg);
  void pointsCallback(const sensor_msgs::PointCloud2ConstPtr& points_msg);
  void globalmapCallback(const sensor_msgs::PointCloud2ConstPtr& points_msg);
  bool relocalize(std_srvs::EmptyRequest& req, std_srvs::EmptyResponse& res);
  void initialposeCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg);
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const;
  void publishOdometry(const ros::Time& stamp, const Eigen::Matrix4f& pose, const double fitness_score);
  void publishScanMatchingStatus(const std_msgs::Header& header, pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned);
  bool getOdomFromTf(nav_msgs::Odometry& odom_out);
  void resetTfBuffer();

private:
  // ROS
  ros::NodeHandle nh_;
  ros::NodeHandle mt_nh_;
  ros::NodeHandle private_nh_;

  std::string global_frame_id_;
  std::string odom_frame_id_;
  std::string base_frame_id_;
  bool tf_broadcast_;
  double cool_time_duration_;
  bool use_odom_;
  bool init_with_tf_;
  ros::Time odom_stamp_last_;

  bool use_imu_;
  bool invert_acc_;
  bool invert_gyro_;
  ros::Subscriber imu_sub_;
  ros::Subscriber points_sub_;
  ros::Subscriber globalmap_sub_;
  ros::Subscriber initialpose_sub_;

  ros::Publisher pose_pub_;
  ros::Publisher aligned_pub_;
  ros::Publisher status_pub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;

  // imu input buffer
  std::mutex imu_data_mutex_;
  std::vector<sensor_msgs::ImuConstPtr> imu_data_;

  // globalmap_ and registration method
  pcl::PointCloud<PointT>::Ptr globalmap_;
  pcl::Filter<PointT>::Ptr downsample_filter_;
  pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr registration_;

  // pose estimator
  std::mutex pose_estimator_mutex_;
  std::unique_ptr<hdl_localization::PoseEstimator> pose_estimator_;

  // global localization
  bool use_global_localization_;
  std::atomic_bool relocalizing_;
  std::unique_ptr<DeltaEstimator> delta_estimator_;

  pcl::PointCloud<PointT>::ConstPtr last_scan_;
  ros::ServiceServer relocalize_server_;
  ros::ServiceClient set_global_map_service_;
  ros::ServiceClient query_global_localization_service_;
};
}  // namespace hdl_localization

#endif  // HDL_LOCALIZATION_NODELET_H
