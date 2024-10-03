#include <nodelet/hdl_localization_nodelet.hpp>

namespace hdl_localization
{
HdlLocalizationNodelet::HdlLocalizationNodelet() : tf_buffer_(), tf_listener_(tf_buffer_)
{
}

HdlLocalizationNodelet::~HdlLocalizationNodelet()
{
}

void HdlLocalizationNodelet::onInit()
{
  nh_ = getNodeHandle();
  mt_nh_ = getMTNodeHandle();
  private_nh_ = getPrivateNodeHandle();
  odom_stamp_last_ = ros::Time::now();

  initializeParams();

  global_frame_id_ = private_nh_.param<std::string>("global_frame_id", "map");
  odom_frame_id_ = private_nh_.param<std::string>("odom_frame_id", "odom");
  base_frame_id_ = private_nh_.param<std::string>("base_frame_id", "base_link");
  tf_broadcast_ = private_nh_.param<bool>("tf_broadcast", true);
  cool_time_duration_ = private_nh_.param<double>("cool_time_duration", 0.0);
  use_odom_ = private_nh_.param<bool>("use_odom", false);
  use_imu_ = private_nh_.param<bool>("use_imu", true);
  if (use_odom_ && use_imu_)
  {
    NODELET_WARN("Both use_odom and use_imu enabled, disabling use_imu");
    use_imu_ = false;
  }
  invert_acc_ = private_nh_.param<bool>("invert_acc", false);
  invert_gyro_ = private_nh_.param<bool>("invert_gyro", false);
  bool specify_init_pose = private_nh_.param<bool>("specify_init_pose", false);
  Eigen::Vector3f init_position = Eigen::Vector3f::Zero();
  Eigen::Quaternionf init_orientation = Eigen::Quaternionf::Identity();
  init_with_tf_ = private_nh_.param<bool>("init_with_tf", true);
  if (specify_init_pose)
  {
    init_position.x() = private_nh_.param<double>("init_pos_x", 0.0);
    init_position.y() = private_nh_.param<double>("init_pos_y", 0.0);
    init_position.z() = private_nh_.param<double>("init_pos_z", 0.0);
    init_orientation.x() = private_nh_.param<double>("init_ori_x", 0.0);
    init_orientation.y() = private_nh_.param<double>("init_ori_y", 0.0);
    init_orientation.z() = private_nh_.param<double>("init_ori_z", 0.0);
    init_orientation.w() = private_nh_.param<double>("init_ori_w", 1.0);
  }
  else if (init_with_tf_)
  {
    if (!tf_broadcast_ && tf_buffer_.canTransform(global_frame_id_, base_frame_id_, ros::Time(0), ros::Duration(2.0)))
    {
      // If global frame and base frame already exist, initial position is set to keep the relative position
      geometry_msgs::TransformStamped tf_map2base =
          tf_buffer_.lookupTransform(global_frame_id_, base_frame_id_, ros::Time(0));
      init_position = Eigen::Vector3f(tf_map2base.transform.translation.x, tf_map2base.transform.translation.y,
                                      tf_map2base.transform.translation.z);
      init_orientation = Eigen::Quaternionf(tf_map2base.transform.rotation.w, tf_map2base.transform.rotation.x,
                                            tf_map2base.transform.rotation.y, tf_map2base.transform.rotation.z);
      NODELET_INFO("global frame(%s) and base frame(%s) exist, use them for initial pose", global_frame_id_.c_str(),
                   base_frame_id_.c_str());
    }
    else if (use_odom_ && tf_buffer_.canTransform(odom_frame_id_, base_frame_id_, ros::Time(0), ros::Duration(2.0)))
    {
      // If there is no global frame, also check the odom frame
      geometry_msgs::TransformStamped tf_odom2base =
          tf_buffer_.lookupTransform(odom_frame_id_, base_frame_id_, ros::Time(0));
      init_position = Eigen::Vector3f(tf_odom2base.transform.translation.x, tf_odom2base.transform.translation.y,
                                      tf_odom2base.transform.translation.z);
      init_orientation = Eigen::Quaternionf(tf_odom2base.transform.rotation.w, tf_odom2base.transform.rotation.x,
                                            tf_odom2base.transform.rotation.y, tf_odom2base.transform.rotation.z);
      NODELET_INFO("global frame(%s) does not exist, use odom frame(%s) instead", global_frame_id_.c_str(),
                   odom_frame_id_.c_str());
    }
    else
    {
      NODELET_WARN("global frame(%s) and odom frame(%s) do not exist, skip initialization", global_frame_id_.c_str(),
                   odom_frame_id_.c_str());
    }
  }
  // Initialize pose estimator
  NODELET_INFO("initialize pose estimator with specified parameters!!");
  pose_estimator_.reset(
      new hdl_localization::PoseEstimator(registration_, init_position, init_orientation, cool_time_duration_));

  // Initialize subscriber and publisher
  if (use_imu_)
  {
    NODELET_INFO("enable imu-based prediction");
    imu_sub_ = mt_nh_.subscribe("/gpsimu_driver/imu_data", 256, &HdlLocalizationNodelet::imuCallback, this);
  }
  points_sub_ = mt_nh_.subscribe("/velodyne_points", 1, &HdlLocalizationNodelet::pointsCallback, this);
  globalmap_sub_ = nh_.subscribe("/globalmap", 1, &HdlLocalizationNodelet::globalmapCallback, this);
  initialpose_sub_ = nh_.subscribe("/initialpose", 1, &HdlLocalizationNodelet::initialposeCallback, this);

  pose_pub_ = nh_.advertise<nav_msgs::Odometry>("/odom", 5, false);
  aligned_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/aligned_points", 5, false);
  status_pub_ = nh_.advertise<ScanMatchingStatus>("/status", 5, false);

  // global localization
  use_global_localization_ = private_nh_.param<bool>("use_global_localization", true);
  if (use_global_localization_)
  {
    NODELET_INFO_STREAM("wait for global localization services");
    ros::service::waitForService("/hdl_global_localization/set_global_map");
    ros::service::waitForService("/hdl_global_localization/query");

    set_global_map_service_ = nh_.serviceClient<hdl_global_localization::SetGlobalMap>("/hdl_global_localization/"
                                                                                       "set_global_map");
    query_global_localization_service_ = nh_.serviceClient<hdl_global_localization::QueryGlobalLocalization>("/hdl_"
                                                                                                             "global_"
                                                                                                             "localiz"
                                                                                                             "ation/"
                                                                                                             "query");

    relocalize_server_ = nh_.advertiseService("/relocalize", &HdlLocalizationNodelet::relocalize, this);
  }
}

pclomp::NormalDistributionsTransform<HdlLocalizationNodelet::PointT, HdlLocalizationNodelet::PointT>::Ptr
HdlLocalizationNodelet::createRegistration() const
{
  std::string reg_method = private_nh_.param<std::string>("reg_method", "NDT_OMP");
  std::string ndt_neighbor_search_method = private_nh_.param<std::string>("ndt_neighbor_search_method", "DIRECT7");
  double ndt_neighbor_search_radius = private_nh_.param<double>("ndt_neighbor_search_radius", 2.0);
  double ndt_resolution = private_nh_.param<double>("ndt_resolution", 1.0);

  if (reg_method == "NDT_OMP")
  {
    NODELET_INFO("NDT_OMP is selected");
    pclomp::NormalDistributionsTransform<HdlLocalizationNodelet::PointT, HdlLocalizationNodelet::PointT>::Ptr ndt(
        new pclomp::NormalDistributionsTransform<HdlLocalizationNodelet::PointT, HdlLocalizationNodelet::PointT>());
    ndt->setTransformationEpsilon(0.01);
    ndt->setResolution(ndt_resolution);
    if (ndt_neighbor_search_method == "DIRECT1")
    {
      NODELET_INFO("search_method DIRECT1 is selected");
      ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
    }
    else if (ndt_neighbor_search_method == "DIRECT7")
    {
      NODELET_INFO("search_method DIRECT7 is selected");
      ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
    }
    else
    {
      if (ndt_neighbor_search_method == "KDTREE")
      {
        NODELET_INFO("search_method KDTREE is selected");
      }
      else
      {
        NODELET_WARN("invalid search method was given");
        NODELET_WARN("default method is selected (KDTREE)");
      }
      ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
    }
    return ndt;
  }

  NODELET_ERROR_STREAM("unknown registration method:" << reg_method);
  return nullptr;
}

void HdlLocalizationNodelet::initializeParams()
{
  // intialize scan matching method
  double downsample_resolution = private_nh_.param<double>("downsample_resolution", 0.1);
  if (downsample_resolution > std::numeric_limits<double>::epsilon())
  {
    boost::shared_ptr<pcl::VoxelGrid<HdlLocalizationNodelet::PointT>> voxelgrid(
        new pcl::VoxelGrid<HdlLocalizationNodelet::PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    downsample_filter_ = voxelgrid;
  }
  else
  {
    ROS_WARN("Input downsample_filter_ is disabled");
    downsample_filter_.reset();
  }

  NODELET_INFO("create registration method for localization");
  registration_ = createRegistration();

  // global localization
  NODELET_INFO("create registration method for fallback during relocalization");
  relocalizing_ = false;
  delta_estimator_.reset(new DeltaEstimator(createRegistration()));
}

/**
 * @brief callback for imu data
 * @param imu_msg
 */
void HdlLocalizationNodelet::imuCallback(const sensor_msgs::ImuConstPtr& imu_msg)
{
  std::lock_guard<std::mutex> lock(imu_data_mutex_);
  imu_data_.push_back(imu_msg);
}

/**
 * @brief callback for point cloud data
 * @param points_msg
 */
void HdlLocalizationNodelet::pointsCallback(const sensor_msgs::PointCloud2ConstPtr& points_msg)
{
  if (!globalmap_)
  {
    NODELET_WARN_THROTTLE(10.0, "Waiting for globalmap");
    return;
  }

  const auto& stamp = points_msg->header.stamp;
  pcl::PointCloud<HdlLocalizationNodelet::PointT>::Ptr pcl_cloud(new pcl::PointCloud<HdlLocalizationNodelet::PointT>());
  pcl::fromROSMsg(*points_msg, *pcl_cloud);

  if (pcl_cloud->empty())
  {
    NODELET_ERROR("cloud is empty!!");
    return;
  }

  ros::Time last_correction_time = pose_estimator_->lastCorrectionTime();
  // Skip calculation if timestamp is wrong
  if (stamp < last_correction_time)
  {
    return;
  }
  // transform pointcloud into base_frame_id
  std::string tf_error;
  pcl::PointCloud<HdlLocalizationNodelet::PointT>::Ptr cloud(new pcl::PointCloud<HdlLocalizationNodelet::PointT>());
  if (this->tf_buffer_.canTransform(base_frame_id_, pcl_cloud->header.frame_id, ros::Time(0), ros::Duration(0.1),
                                    &tf_error))
  {
    if (!pcl_ros::transformPointCloud(base_frame_id_, *pcl_cloud, *cloud, this->tf_buffer_))
    {
      NODELET_ERROR("point cloud cannot be transformed into target frame (%s -> %s)",
                    pcl_cloud->header.frame_id.c_str(), base_frame_id_.c_str());
      return;
    }
  }
  else
  {
    NODELET_ERROR_STREAM("TF transform failed," << tf_error.c_str());
    return;
  }

  auto filtered = downsample(cloud);
  last_scan_ = filtered;

  if (relocalizing_)
  {
    delta_estimator_->addFrame(filtered);
  }

  std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex_);
  if (!pose_estimator_)
  {
    NODELET_ERROR_THROTTLE(10.0, "Waiting for initial pose input");
    return;
  }
  Eigen::Matrix4f before = pose_estimator_->matrix();

  if (use_imu_)
  {
    // PointClouds + IMU prediction
    if (imu_data_.empty())
    {
      NODELET_WARN_THROTTLE(10.0, "Waiting for imu data");
      return;
    }
    std::lock_guard<std::mutex> lock(imu_data_mutex_);
    auto imu_iter = imu_data_.begin();
    for (imu_iter; imu_iter != imu_data_.end(); imu_iter++)
    {
      if (stamp < (*imu_iter)->header.stamp)
      {
        break;
      }
      const auto& acc = (*imu_iter)->linear_acceleration;
      const auto& gyro = (*imu_iter)->angular_velocity;
      double acc_sign = invert_acc_ ? -1.0 : 1.0;
      double gyro_sign = invert_gyro_ ? -1.0 : 1.0;
      pose_estimator_->predictImu((*imu_iter)->header.stamp, acc_sign * Eigen::Vector3f(acc.x, acc.y, acc.z),
                                  gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
    }
    imu_data_.erase(imu_data_.begin(), imu_iter);
  }
  else if (use_odom_)
  {
    // PointClouds + Odometry prediction
    nav_msgs::Odometry odom;
    if (!getOdomFromTf(odom))
    {
      pose_estimator_->predict(stamp);
      return;
    }
    Eigen::Vector3f odom_twist_linear(odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z);
    Eigen::Vector3f odom_twist_angular(odom.twist.twist.angular.x, odom.twist.twist.angular.y,
                                       odom.twist.twist.angular.z);
    pose_estimator_->predictOdom(odom.header.stamp, odom_twist_linear, odom_twist_angular);
  }
  else
  {
    // PointClouds-only prediction
    pose_estimator_->predict(stamp);
  }
  // Perform scan matching using the calculated position as the initial value
  double fitness_score;
  auto aligned = pose_estimator_->correct(stamp, filtered, fitness_score);

  if (aligned_pub_.getNumSubscribers())
  {
    aligned->header.frame_id = global_frame_id_;
    aligned->header.stamp = cloud->header.stamp;
    aligned_pub_.publish(aligned);
  }

  if (status_pub_.getNumSubscribers())
  {
    publishScanMatchingStatus(points_msg->header, aligned);
  }

  publishOdometry(points_msg->header.stamp, pose_estimator_->matrix(), fitness_score);
}

/**
 * @brief callback for globalmap input
 * @param points_msg
 */
void HdlLocalizationNodelet::globalmapCallback(const sensor_msgs::PointCloud2ConstPtr& points_msg)
{
  NODELET_INFO("Globalmap received");
  pcl::PointCloud<HdlLocalizationNodelet::PointT>::Ptr cloud(new pcl::PointCloud<HdlLocalizationNodelet::PointT>());
  pcl::fromROSMsg(*points_msg, *cloud);
  globalmap_ = cloud;

  registration_->setInputTarget(globalmap_);

  if (use_global_localization_)
  {
    NODELET_INFO("set globalmap for global localization!");
    hdl_global_localization::SetGlobalMap srv;
    pcl::toROSMsg(*globalmap_, srv.request.global_map);

    if (!set_global_map_service_.call(srv))
    {
      NODELET_INFO("failed to set global map");
    }
    else
    {
      NODELET_INFO("done");
    }
  }
}

/**
 * @brief perform global localization to relocalize the sensor position
 * @param
 */
bool HdlLocalizationNodelet::relocalize(std_srvs::EmptyRequest& /*req*/, std_srvs::EmptyResponse& /*res*/)
{
  if (last_scan_ == nullptr)
  {
    NODELET_INFO_STREAM("no scan has been received");
    return false;
  }

  relocalizing_ = true;
  delta_estimator_->reset();
  pcl::PointCloud<HdlLocalizationNodelet::PointT>::ConstPtr scan = last_scan_;

  hdl_global_localization::QueryGlobalLocalization srv;
  pcl::toROSMsg(*scan, srv.request.cloud);
  srv.request.max_num_candidates = 1;

  if (!query_global_localization_service_.call(srv) || srv.response.poses.empty())
  {
    relocalizing_ = false;
    NODELET_INFO_STREAM("global localization failed");
    return false;
  }

  const auto& result = srv.response.poses[0];

  NODELET_INFO_STREAM("--- Global localization result ---");
  NODELET_INFO_STREAM("Trans :" << result.position.x << " " << result.position.y << " " << result.position.z);
  NODELET_INFO_STREAM("Quat  :" << result.orientation.x << " " << result.orientation.y << " " << result.orientation.z
                                << " " << result.orientation.w);
  NODELET_INFO_STREAM("Error :" << srv.response.errors[0]);
  NODELET_INFO_STREAM("Inlier:" << srv.response.inlier_fractions[0]);

  Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  pose.linear() =
      Eigen::Quaternionf(result.orientation.w, result.orientation.x, result.orientation.y, result.orientation.z)
          .toRotationMatrix();
  pose.translation() = Eigen::Vector3f(result.position.x, result.position.y, result.position.z);
  pose = pose * delta_estimator_->estimatedDelta();

  std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
  pose_estimator_.reset(new hdl_localization::PoseEstimator(registration_, pose.translation(),
                                                            Eigen::Quaternionf(pose.linear()), cool_time_duration_));

  relocalizing_ = false;

  return true;
}

/**
 * @brief callback for initial pose input ("2D Pose Estimate" on rviz)
 * @param pose_msg
 */
void HdlLocalizationNodelet::initialposeCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg)
{
  NODELET_INFO("initialpose received");
  // initialize tf_global2rviz on origin
  geometry_msgs::TransformStamped tf_global2rviz;
  tf_global2rviz.transform.translation.x = 0.0;
  tf_global2rviz.transform.translation.y = 0.0;
  tf_global2rviz.transform.translation.z = 0.0;
  tf_global2rviz.transform.rotation.x = 0.0;
  tf_global2rviz.transform.rotation.y = 0.0;
  tf_global2rviz.transform.rotation.z = 0.0;
  tf_global2rviz.transform.rotation.w = 1.0;
  // If the origin of rviz and the global(map) frame are different, coordinate transformation is performed
  if (init_with_tf_ && global_frame_id_ != pose_msg->header.frame_id && !global_frame_id_.empty() &&
      !pose_msg->header.frame_id.empty())
  {
    tf_global2rviz =
        tf_buffer_.lookupTransform(global_frame_id_, pose_msg->header.frame_id, ros::Time(0), ros::Duration(2.0));
    if (tf_global2rviz.header.frame_id != global_frame_id_)
    {
      ROS_ERROR("[Failed to get transform from %s to %s", global_frame_id_.c_str(), pose_msg->header.frame_id.c_str());
    }
  }
  tf2::Vector3 vec_rviz2robot(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y,
                              pose_msg->pose.pose.position.z);
  tf2::Quaternion q_rviz2robot(pose_msg->pose.pose.orientation.x, pose_msg->pose.pose.orientation.y,
                               pose_msg->pose.pose.orientation.z, pose_msg->pose.pose.orientation.w);
  // Get initial pose from global to robot
  tf2::Vector3 vec_global2rviz(tf_global2rviz.transform.translation.x, tf_global2rviz.transform.translation.y,
                               tf_global2rviz.transform.translation.z);
  tf2::Quaternion q_global2rviz(tf_global2rviz.transform.rotation.x, tf_global2rviz.transform.rotation.y,
                                tf_global2rviz.transform.rotation.z, tf_global2rviz.transform.rotation.w);
  tf2::Vector3 vec_global2robot = vec_global2rviz + (tf2::Matrix3x3(q_global2rviz) * vec_rviz2robot);
  tf2::Quaternion q_global2robot = q_global2rviz * q_rviz2robot;
  Eigen::Vector3f init_position = Eigen::Vector3f(vec_global2robot.x(), vec_global2robot.y(), vec_global2robot.z());
  Eigen::Quaternionf init_orientation =
      Eigen::Quaternionf(q_global2robot.w(), q_global2robot.x(), q_global2robot.y(), q_global2robot.z());
  if (use_odom_)
  {
    resetTfBuffer();
  }
  if (use_imu_)
  {
    imu_data_.clear();
  }
  std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
  pose_estimator_.reset(
      new hdl_localization::PoseEstimator(registration_, init_position, init_orientation, cool_time_duration_));
}

/**
 * @brief downsampling
 * @param cloud   input cloud
 * @return downsampled cloud
 */
pcl::PointCloud<HdlLocalizationNodelet::PointT>::ConstPtr
HdlLocalizationNodelet::downsample(const pcl::PointCloud<HdlLocalizationNodelet::PointT>::ConstPtr& cloud) const
{
  if (!downsample_filter_)
  {
    return cloud;
  }

  pcl::PointCloud<HdlLocalizationNodelet::PointT>::Ptr filtered(new pcl::PointCloud<HdlLocalizationNodelet::PointT>());
  downsample_filter_->setInputCloud(cloud);
  downsample_filter_->filter(*filtered);
  filtered->header = cloud->header;

  return filtered;
}

/**
 * @brief publish odometry
 * @param stamp  timestamp
 * @param pose   odometry pose to be published
 */
void HdlLocalizationNodelet::publishOdometry(const ros::Time& stamp, const Eigen::Matrix4f& pose,
                                             const double fitness_score)
{
  // broadcast the transform over tf
  if (tf_broadcast_)
  {
    if (tf_buffer_.canTransform(odom_frame_id_, base_frame_id_, ros::Time(0)))
    {
      geometry_msgs::TransformStamped map_wrt_frame =
          tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
      map_wrt_frame.header.stamp = stamp;
      map_wrt_frame.header.frame_id = base_frame_id_;
      map_wrt_frame.child_frame_id = global_frame_id_;

      geometry_msgs::TransformStamped frame_wrt_odom =
          tf_buffer_.lookupTransform(odom_frame_id_, base_frame_id_, ros::Time(0), ros::Duration(0.1));
      Eigen::Matrix4f frame2odom = tf2::transformToEigen(frame_wrt_odom).cast<float>().matrix();

      geometry_msgs::TransformStamped map_wrt_odom;
      tf2::doTransform(map_wrt_frame, map_wrt_odom, frame_wrt_odom);

      tf2::Transform odom_wrt_map;
      tf2::fromMsg(map_wrt_odom.transform, odom_wrt_map);
      odom_wrt_map = odom_wrt_map.inverse();

      geometry_msgs::TransformStamped odom_trans;
      odom_trans.transform = tf2::toMsg(odom_wrt_map);
      odom_trans.header.stamp = stamp;
      odom_trans.header.frame_id = global_frame_id_;
      odom_trans.child_frame_id = odom_frame_id_;

      tf_broadcaster_.sendTransform(odom_trans);
    }
    else
    {
      geometry_msgs::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
      odom_trans.header.stamp = stamp;
      odom_trans.header.frame_id = global_frame_id_;
      odom_trans.child_frame_id = base_frame_id_;
      tf_broadcaster_.sendTransform(odom_trans);
    }
  }

  // publish the transform
  nav_msgs::Odometry odom;
  odom.header.stamp = stamp;
  odom.header.frame_id = global_frame_id_;

  tf::poseEigenToMsg(Eigen::Isometry3d(pose.cast<double>()), odom.pose.pose);
  odom.pose.covariance[0] = private_nh_.param<double>("cov_scale_x", 1.0) * fitness_score;
  odom.pose.covariance[7] = private_nh_.param<double>("cov_scale_y", 1.0) * fitness_score;
  odom.pose.covariance[14] = private_nh_.param<double>("cov_scale_z", 1.0) * fitness_score;
  odom.pose.covariance[21] = private_nh_.param<double>("cov_scale_roll", 1.0) * fitness_score;
  odom.pose.covariance[28] = private_nh_.param<double>("cov_scale_pitch", 1.0) * fitness_score;
  odom.pose.covariance[35] = private_nh_.param<double>("cov_scale_yaw", 1.0) * fitness_score;

  odom.child_frame_id = base_frame_id_;
  odom.twist.twist.linear.x = 0.0;
  odom.twist.twist.linear.y = 0.0;
  odom.twist.twist.angular.z = 0.0;

  pose_pub_.publish(odom);
}

/**
 * @brief publish scan matching status information
 */
void HdlLocalizationNodelet::publishScanMatchingStatus(const std_msgs::Header& header,
                                                       pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned)
{
  ScanMatchingStatus status;
  status.header = header;

  status.has_converged = registration_->hasConverged();
  status.matching_error = 0.0;

  const double max_correspondence_dist = private_nh_.param<double>("status_max_correspondence_dist", 0.5);
  const double max_valid_point_dist = private_nh_.param<double>("status_max_valid_point_dist", 25.0);

  int num_inliers = 0;
  int num_valid_points = 0;
  std::vector<int> k_indices;
  std::vector<float> k_sq_dists;
  for (int i = 0; i < aligned->size(); i++)
  {
    const auto& pt = aligned->at(i);
    if (pt.getVector3fMap().norm() > max_valid_point_dist)
    {
      continue;
    }
    num_valid_points++;

    registration_->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
    if (k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist)
    {
      status.matching_error += k_sq_dists[0];
      num_inliers++;
    }
  }

  status.matching_error /= num_inliers;
  status.inlier_fraction = static_cast<float>(num_inliers) / std::max(1, num_valid_points);
  status.relative_pose =
      tf2::eigenToTransform(Eigen::Isometry3d(registration_->getFinalTransformation().cast<double>())).transform;

  status.prediction_labels.reserve(2);
  status.prediction_errors.reserve(2);

  std::vector<double> errors(6, 0.0);

  if (pose_estimator_->woPredictionError())
  {
    status.prediction_labels.push_back(std_msgs::String());
    status.prediction_labels.back().data = "without_pred";
    status.prediction_errors.push_back(
        tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator_->woPredictionError().get().cast<double>())).transform);
  }

  if (pose_estimator_->motionPredictionError())
  {
    status.prediction_labels.push_back(std_msgs::String());
    if (use_imu_)
    {
      status.prediction_labels.back().data = "imu";
    }
    else if (use_odom_)
    {
      status.prediction_labels.back().data = "odom";
    }
    else
    {
      status.prediction_labels.back().data = "motion_model";
    }

    status.prediction_errors.push_back(
        tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator_->motionPredictionError().get().cast<double>()))
            .transform);
  }
  status_pub_.publish(status);
}

bool HdlLocalizationNodelet::getOdomFromTf(nav_msgs::Odometry& odom_out)
{
  double initialize_timeout = 0.1;
  if (odom_frame_id_.empty() || base_frame_id_.empty())
  {
    NODELET_WARN_THROTTLE(1.0, "odom_frame_id(%s) or base_frame_id(%s) is not set", odom_frame_id_.c_str(),
                          base_frame_id_.c_str());
    return false;
  }
  if (tf_buffer_.canTransform(base_frame_id_, odom_stamp_last_, base_frame_id_, ros::Time(0), odom_frame_id_,
                              ros::Duration(initialize_timeout)) &&
      tf_buffer_.canTransform(odom_frame_id_, base_frame_id_, ros::Time(0), ros::Duration(initialize_timeout)))
  {
    // Get the amount of odometry movement since the last calculation
    // Coordinate system where the front of the robot is x
    geometry_msgs::TransformStamped odom_delta =
        tf_buffer_.lookupTransform(base_frame_id_, odom_stamp_last_, base_frame_id_, ros::Time(0), odom_frame_id_);
    // Get the latest base_frame_ to get the time
    geometry_msgs::TransformStamped odom_now = tf_buffer_.lookupTransform(odom_frame_id_, base_frame_id_, ros::Time(0));
    ros::Time odom_stamp = odom_now.header.stamp;
    double odom_time_diff = (odom_stamp - odom_stamp_last_).toSec();
    if (odom_delta.header.stamp.isZero() || odom_time_diff < std::numeric_limits<double>::epsilon())
    {
      NODELET_WARN("Wrong timestamp detected: odom_time_diff = %f", odom_time_diff);
      odom_stamp_last_ = odom_stamp;
      return false;
    }
    else
    {
      Eigen::Vector3f odom_travel_linear(odom_delta.transform.translation.x, odom_delta.transform.translation.y,
                                         odom_delta.transform.translation.z);
      Eigen::Vector3f odom_twist_linear = odom_travel_linear / odom_time_diff;
      tf::Quaternion odom_travel_angular(odom_delta.transform.rotation.x, odom_delta.transform.rotation.y,
                                         odom_delta.transform.rotation.z, odom_delta.transform.rotation.w);
      double roll, pitch, yaw;
      tf::Matrix3x3(odom_travel_angular).getRPY(roll, pitch, yaw);
      Eigen::Vector3f odom_twist_angular(roll / odom_time_diff, pitch / odom_time_diff, yaw / odom_time_diff);
      odom_out.header.stamp = odom_stamp;
      odom_out.header.frame_id = odom_frame_id_;
      odom_out.child_frame_id = base_frame_id_;
      odom_out.pose.pose.position.x = odom_now.transform.translation.x;
      odom_out.pose.pose.position.y = odom_now.transform.translation.y;
      odom_out.pose.pose.position.z = odom_now.transform.translation.z;
      odom_out.pose.pose.orientation.x = odom_now.transform.rotation.x;
      odom_out.pose.pose.orientation.y = odom_now.transform.rotation.y;
      odom_out.pose.pose.orientation.z = odom_now.transform.rotation.z;
      odom_out.pose.pose.orientation.w = odom_now.transform.rotation.w;
      odom_out.twist.twist.linear.x = odom_twist_linear.x();
      odom_out.twist.twist.linear.y = odom_twist_linear.y();
      odom_out.twist.twist.linear.z = odom_twist_linear.z();
      odom_out.twist.twist.angular.x = odom_twist_angular.x();
      odom_out.twist.twist.angular.y = odom_twist_angular.y();
      odom_out.twist.twist.angular.z = odom_twist_angular.z();
    }
    odom_stamp_last_ = odom_stamp;
    return true;
  }
  else
  {
    if (tf_buffer_.canTransform(odom_frame_id_, base_frame_id_, ros::Time(0), ros::Duration(initialize_timeout)))
    {
      NODELET_INFO("Received the first odom transform");
      // Get the latest base_frame_ to get the time
      geometry_msgs::TransformStamped odom_now =
          tf_buffer_.lookupTransform(odom_frame_id_, base_frame_id_, ros::Time(0));
      odom_stamp_last_ = odom_now.header.stamp;
      return false;
    }
    else
    {
      NODELET_WARN_STREAM("Failed to look up transform between " << odom_frame_id_ << " and " << odom_frame_id_);
      return false;
    }
  }
  return false;
}

void HdlLocalizationNodelet::resetTfBuffer()
{
  tf_buffer_.clear();
  odom_stamp_last_ = ros::Time(0);
}

}  // namespace hdl_localization

PLUGINLIB_EXPORT_CLASS(hdl_localization::HdlLocalizationNodelet, nodelet::Nodelet)
