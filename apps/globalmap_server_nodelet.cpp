#define PCL_NO_PRECOMPILE

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

  GlobalmapServerNodelet()
  {
  }
  ~GlobalmapServerNodelet() override
  {
  }

  void onInit() override
  {
    nh_ = getNodeHandle();
    mt_nh_ = getMTNodeHandle();
    private_nh_ = getPrivateNodeHandle();

    initializeParams();

    // publish globalmap with "latched" publisher
    globalmap_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/globalmap", 5, true);
    map_update_sub_ = nh_.subscribe("/map_request/pcd", 10, &GlobalmapServerNodelet::mapUpdateCallback, this);

    globalmap_pub_timer_ =
        nh_.createWallTimer(ros::WallDuration(1.0), &GlobalmapServerNodelet::pubOnceCb, this, true, true);
  }

private:
  void initializeParams()
  {
    // read globalmap from a pcd file
    std::string globalmap_pcd = private_nh_.param<std::string>("globalmap_pcd", "");
    globalmap_.reset(new pcl::PointCloud<PointT>());
    pcl::io::loadPCDFile(globalmap_pcd, *globalmap_);
    globalmap_->header.frame_id = "map";

    std::ifstream utm_file(globalmap_pcd + ".utm");
    if (utm_file.is_open() && private_nh_.param<bool>("convert_utm_to_local", true))
    {
      double utm_easting;
      double utm_northing;
      double altitude;
      utm_file >> utm_easting >> utm_northing >> altitude;
      for (auto& pt : globalmap_->points)
      {
        pt.getVector3fMap() -= Eigen::Vector3f(utm_easting, utm_northing, altitude);
      }
      ROS_INFO_STREAM("Global map offset by UTM reference coordinates (x = "
                      << utm_easting << ", y = " << utm_northing << ") and altitude (z = " << altitude << ")");
    }

    // downsample globalmap
    double downsample_resolution = private_nh_.param<double>("downsample_resolution", 0.1);
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    voxelgrid->setInputCloud(globalmap_);

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    voxelgrid->filter(*filtered);

    globalmap_ = filtered;
  }

  void pubOnceCb(const ros::WallTimerEvent& event)
  {
    globalmap_pub_.publish(globalmap_);
  }

  void mapUpdateCallback(const std_msgs::String& msg)
  {
    ROS_INFO_STREAM("Received map request, map path : " << msg.data);
    std::string globalmap_pcd = msg.data;
    globalmap_.reset(new pcl::PointCloud<PointT>());
    pcl::io::loadPCDFile(globalmap_pcd, *globalmap_);
    globalmap_->header.frame_id = "map";

    // downsample globalmap
    double downsample_resolution = private_nh_.param<double>("downsample_resolution", 0.1);
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    voxelgrid->setInputCloud(globalmap_);

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    voxelgrid->filter(*filtered);

    globalmap_ = filtered;
    globalmap_pub_.publish(globalmap_);
  }

private:
  // ROS
  ros::NodeHandle nh_;
  ros::NodeHandle mt_nh_;
  ros::NodeHandle private_nh_;

  ros::Publisher globalmap_pub_;
  ros::Subscriber map_update_sub_;

  ros::WallTimer globalmap_pub_timer_;
  pcl::PointCloud<PointT>::Ptr globalmap_;
};

}  // namespace hdl_localization

PLUGINLIB_EXPORT_CLASS(hdl_localization::GlobalmapServerNodelet, nodelet::Nodelet)
