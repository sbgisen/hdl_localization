#define PCL_NO_PRECOMPILE
#include <nodelet/globalmap_server_nodelet.hpp>

namespace hdl_localization
{
GlobalmapServerNodelet::GlobalmapServerNodelet()
{
}

GlobalmapServerNodelet::~GlobalmapServerNodelet()
{
}

void GlobalmapServerNodelet::onInit()
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

void GlobalmapServerNodelet::initializeParams()
{
  // read globalmap from a pcd file
  std::string globalmap_pcd = private_nh_.param<std::string>("globalmap_pcd", "");
  global_frame_id_ = private_nh_.param<std::string>("global_frame_id", "map");
  globalmap_.reset(new pcl::PointCloud<PointT>());
  pcl::io::loadPCDFile(globalmap_pcd, *globalmap_);
  globalmap_->header.frame_id = global_frame_id_;

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

  double downsample_resolution = private_nh_.param<double>("downsample_resolution", 0.1);
  if (downsample_resolution > std::numeric_limits<double>::epsilon())
  {
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    voxelgrid->setInputCloud(globalmap_);
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    voxelgrid->filter(*filtered);
    globalmap_ = filtered;
  }
  else
  {
    ROS_WARN_STREAM("Globalmap will not be downsampled");
  }
}

void GlobalmapServerNodelet::pubOnceCb(const ros::WallTimerEvent& event)
{
  globalmap_pub_.publish(globalmap_);
}

void GlobalmapServerNodelet::mapUpdateCallback(const std_msgs::String& msg)
{
  ROS_INFO_STREAM("Received map request, map path : " << msg.data);
  std::string globalmap_pcd = msg.data;
  globalmap_.reset(new pcl::PointCloud<PointT>());
  pcl::io::loadPCDFile(globalmap_pcd, *globalmap_);
  globalmap_->header.frame_id = global_frame_id_;
  // downsample globalmap
  double downsample_resolution = private_nh_.param<double>("downsample_resolution", 0.1);
  if (downsample_resolution > std::numeric_limits<double>::epsilon())
  {
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    voxelgrid->setInputCloud(globalmap_);
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    voxelgrid->filter(*filtered);
    globalmap_ = filtered;
  }
  else
  {
    ROS_WARN("Globalmap will not be downsampled");
  }
  globalmap_pub_.publish(globalmap_);
}

}  // namespace hdl_localization

PLUGINLIB_EXPORT_CLASS(hdl_localization::GlobalmapServerNodelet, nodelet::Nodelet)
