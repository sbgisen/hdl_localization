#include <mutex>
#include <memory>
#include <iostream>

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
#include <hdl_localization/delta_estimater.hpp>

#include <hdl_localization/ScanMatchingStatus.h>
#include <hdl_global_localization/SetGlobalMap.h>
#include <hdl_global_localization/QueryGlobalLocalization.h>

namespace hdl_localization
{
class HdlLocalizationNodelet : public nodelet::Nodelet
{
public:
  using PointT = pcl::PointXYZI;

  HdlLocalizationNodelet() : tf_buffer_(), tf_listener_(tf_buffer_)
  {
  }
  ~HdlLocalizationNodelet() override
  {
  }

  void onInit() override
  {
    nh_ = getNodeHandle();
    mt_nh_ = getMTNodeHandle();
    private_nh_ = getPrivateNodeHandle();
    odom_stamp_last_ = ros::Time::now();

    initializeParams();

    use_odom_ = private_nh_.param<bool>("enable_robot_odometry_prediction", false);
    robot_odom_frame_id_ = private_nh_.param<std::string>("robot_odom_frame_id", "robot_odom");
    odom_child_frame_id_ = private_nh_.param<std::string>("odom_child_frame_id", "base_link");
    enable_tf_ = private_nh_.param<bool>("enable_tf", true);

    use_imu_ = private_nh_.param<bool>("use_imu", true);
    if (use_odom_ && use_imu_)
    {
      NODELET_WARN("[HdlLocalizationNodelet] Both use_odom and use_imu enabled -> disabling use_imu");
      use_imu_ = false;
    }
    invert_acc_ = private_nh_.param<bool>("invert_acc", false);
    invert_gyro_ = private_nh_.param<bool>("invert_gyro", false);
    if (use_imu_)
    {
      NODELET_INFO("enable imu-based prediction");
      imu_sub_ = mt_nh_.subscribe("/gpsimu_driver/imu_data", 256, &HdlLocalizationNodelet::imuCallback, this);
    }
    points_sub_ = mt_nh_.subscribe("/velodyne_points", 5, &HdlLocalizationNodelet::pointsCallback, this);
    globalmap_sub_ = nh_.subscribe("/globalmap", 1, &HdlLocalizationNodelet::globalmapCallback, this);
    initialpose_sub_ = nh_.subscribe("/initialpose", 8, &HdlLocalizationNodelet::initialposeCallback, this);

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

private:
  pcl::Registration<PointT, PointT>::Ptr createRegistration() const
  {
    std::string reg_method = private_nh_.param<std::string>("reg_method", "NDT_OMP");
    std::string ndt_neighbor_search_method = private_nh_.param<std::string>("ndt_neighbor_search_method", "DIRECT7");
    double ndt_neighbor_search_radius = private_nh_.param<double>("ndt_neighbor_search_radius", 2.0);
    double ndt_resolution = private_nh_.param<double>("ndt_resolution", 1.0);

    if (reg_method == "NDT_OMP")
    {
      NODELET_INFO("NDT_OMP is selected");
      pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(
          new pclomp::NormalDistributionsTransform<PointT, PointT>());
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
    else if (reg_method.find("NDT_CUDA") != std::string::npos)
    {
      NODELET_INFO("NDT_CUDA is selected");
      boost::shared_ptr<fast_gicp::NDTCuda<PointT, PointT>> ndt(new fast_gicp::NDTCuda<PointT, PointT>);
      ndt->setResolution(ndt_resolution);

      if (reg_method.find("D2D") != std::string::npos)
      {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::D2D);
      }
      else if (reg_method.find("P2D") != std::string::npos)
      {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
      }

      if (ndt_neighbor_search_method == "DIRECT1")
      {
        NODELET_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
      }
      else if (ndt_neighbor_search_method == "DIRECT7")
      {
        NODELET_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
      }
      else if (ndt_neighbor_search_method == "DIRECT_RADIUS")
      {
        NODELET_INFO_STREAM("search_method DIRECT_RADIUS is selected : " << ndt_neighbor_search_radius);
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT_RADIUS, ndt_neighbor_search_radius);
      }
      else
      {
        NODELET_WARN("invalid search method was given");
      }
      return ndt;
    }

    NODELET_ERROR_STREAM("unknown registration method:" << reg_method);
    return nullptr;
  }

  void initializeParams()
  {
    // intialize scan matching method
    double downsample_resolution = private_nh_.param<double>("downsample_resolution", 0.1);
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    downsample_filter_ = voxelgrid;

    NODELET_INFO("create registration method for localization");
    registration_ = createRegistration();

    // global localization
    NODELET_INFO("create registration method for fallback during relocalization");
    relocalizing_ = false;
    delta_estimater_.reset(new DeltaEstimater(createRegistration()));

    // initialize pose estimator
    if (private_nh_.param<bool>("specify_init_pose", true))
    {
      NODELET_INFO("initialize pose estimator with specified parameters!!");
      pose_estimator_.reset(new hdl_localization::PoseEstimator(
          registration_,
          Eigen::Vector3f(private_nh_.param<double>("init_pos_x", 0.0), private_nh_.param<double>("init_pos_y", 0.0),
                          private_nh_.param<double>("init_pos_z", 0.0)),
          Eigen::Quaternionf(private_nh_.param<double>("init_ori_w", 1.0), private_nh_.param<double>("init_ori_x", 0.0),
                             private_nh_.param<double>("init_ori_y", 0.0),
                             private_nh_.param<double>("init_ori_z", 0.0)),
          private_nh_.param<double>("cool_time_duration", 0.5)));
    }
  }

private:
  /**
   * @brief callback for imu data
   * @param imu_msg
   */
  void imuCallback(const sensor_msgs::ImuConstPtr& imu_msg)
  {
    std::lock_guard<std::mutex> lock(imu_data_mutex_);
    imu_data_.push_back(imu_msg);
  }

  /**
   * @brief callback for point cloud data
   * @param points_msg
   */
  void pointsCallback(const sensor_msgs::PointCloud2ConstPtr& points_msg)
  {
    if (!globalmap_)
    {
      NODELET_ERROR("globalmap has not been received!!");
      return;
    }

    const auto& stamp = points_msg->header.stamp;
    pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
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
    // transform pointcloud into odom_child_frame_id
    std::string tf_error;
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if (this->tf_buffer_.canTransform(odom_child_frame_id_, pcl_cloud->header.frame_id, stamp, ros::Duration(0.1),
                                      &tf_error))
    {
      if (!pcl_ros::transformPointCloud(odom_child_frame_id_, *pcl_cloud, *cloud, this->tf_buffer_))
      {
        NODELET_ERROR("point cloud cannot be transformed into target frame!!");
        return;
      }
    }
    else
    {
      NODELET_ERROR_STREAM(tf_error.c_str());
      return;
    }

    auto filtered = downsample(cloud);
    last_scan_ = filtered;

    if (relocalizing_)
    {
      delta_estimater_->addFrame(filtered);
    }

    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex_);
    if (!pose_estimator_)
    {
      NODELET_ERROR("waiting for initial pose input!!");
      return;
    }
    Eigen::Matrix4f before = pose_estimator_->matrix();

    if (use_imu_)
    {
      // PointClouds + IMU prediction
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
      // PointClouds + Oodometry prediction
      if (tf_buffer_.canTransform(odom_child_frame_id_, odom_stamp_last_, odom_child_frame_id_, ros::Time(0),
                                  robot_odom_frame_id_, ros::Duration(0)))
      {
        // Get the amount of odometry movement since the last calculation
        // Coordinate system where the front of the robot is x
        geometry_msgs::TransformStamped odom_delta =
            tf_buffer_.lookupTransform(odom_child_frame_id_, odom_stamp_last_, odom_child_frame_id_, ros::Time(0),
                                       robot_odom_frame_id_, ros::Duration(0));
        // Get the latest odom_child_frame_id to get the time
        geometry_msgs::TransformStamped odom_now =
            tf_buffer_.lookupTransform(robot_odom_frame_id_, odom_child_frame_id_, ros::Time(0));
        ros::Time odom_stamp = odom_now.header.stamp;
        ros::Duration odom_time_diff = odom_stamp - odom_stamp_last_;
        double odom_time_diff_sec = odom_time_diff.toSec();
        if (odom_delta.header.stamp.isZero() || odom_time_diff_sec <= 0)
        {
          NODELET_WARN_STREAM("Wrong timestamp detected: odom_time_diff_sec = " << odom_time_diff_sec);
        }
        else
        {
          Eigen::Vector3f odom_travel_linear(odom_delta.transform.translation.x, odom_delta.transform.translation.y,
                                             odom_delta.transform.translation.z);
          Eigen::Vector3f odom_twist_linear = odom_travel_linear / odom_time_diff_sec;
          tf::Quaternion odom_travel_angular(odom_delta.transform.rotation.x, odom_delta.transform.rotation.y,
                                             odom_delta.transform.rotation.z, odom_delta.transform.rotation.w);
          double roll, pitch, yaw;
          tf::Matrix3x3(odom_travel_angular).getRPY(roll, pitch, yaw);
          Eigen::Vector3f odom_twist_angular(roll / odom_time_diff_sec, pitch / odom_time_diff_sec,
                                             yaw / odom_time_diff_sec);
          pose_estimator_->predictOdom(odom_stamp, odom_twist_linear, odom_twist_angular);
        }
        odom_stamp_last_ = odom_stamp;
      }
      else
      {
        if (tf_buffer_.canTransform(robot_odom_frame_id_, odom_child_frame_id_, ros::Time(0)))
        {
          NODELET_WARN_STREAM("The last timestamp is wrong, skip localization");
          // Get the latest odom_child_frame_id to get the time
          geometry_msgs::TransformStamped odom_now =
              tf_buffer_.lookupTransform(robot_odom_frame_id_, odom_child_frame_id_, ros::Time(0));
          odom_stamp_last_ = odom_now.header.stamp;
        }
        else
        {
          NODELET_WARN_STREAM("Failed to look up transform between " << cloud->header.frame_id << " and "
                                                                     << robot_odom_frame_id_);
        }
      }
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
      aligned->header.frame_id = "map";
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
  void globalmapCallback(const sensor_msgs::PointCloud2ConstPtr& points_msg)
  {
    NODELET_INFO("globalmap received!");
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
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
  bool relocalize(std_srvs::EmptyRequest& /*req*/, std_srvs::EmptyResponse& /*res*/)
  {
    if (last_scan_ == nullptr)
    {
      NODELET_INFO_STREAM("no scan has been received");
      return false;
    }

    relocalizing_ = true;
    delta_estimater_->reset();
    pcl::PointCloud<PointT>::ConstPtr scan = last_scan_;

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
    pose = pose * delta_estimater_->estimatedDelta();

    std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
    pose_estimator_.reset(new hdl_localization::PoseEstimator(registration_, pose.translation(),
                                                              Eigen::Quaternionf(pose.linear()),
                                                              private_nh_.param<double>("cool_time_duration", 0.5)));

    relocalizing_ = false;

    return true;
  }

  /**
   * @brief callback for initial pose input ("2D Pose Estimate" on rviz)
   * @param pose_msg
   */
  void initialposeCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg)
  {
    NODELET_INFO("initial pose received!!");
    std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
    const auto& p = pose_msg->pose.pose.position;
    const auto& q = pose_msg->pose.pose.orientation;
    pose_estimator_.reset(new hdl_localization::PoseEstimator(registration_, Eigen::Vector3f(p.x, p.y, p.z),
                                                              Eigen::Quaternionf(q.w, q.x, q.y, q.z),
                                                              private_nh_.param<double>("cool_time_duration", 0.5)));
  }

  /**
   * @brief downsampling
   * @param cloud   input cloud
   * @return downsampled cloud
   */
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const
  {
    if (!downsample_filter_)
    {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
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
  void publishOdometry(const ros::Time& stamp, const Eigen::Matrix4f& pose, const double fitness_score)
  {
    // broadcast the transform over tf
    if (enable_tf_)
    {
      if (tf_buffer_.canTransform(robot_odom_frame_id_, odom_child_frame_id_, ros::Time(0)))
      {
        geometry_msgs::TransformStamped map_wrt_frame =
            tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
        map_wrt_frame.header.stamp = stamp;
        map_wrt_frame.header.frame_id = odom_child_frame_id_;
        map_wrt_frame.child_frame_id = "map";

        geometry_msgs::TransformStamped frame_wrt_odom =
            tf_buffer_.lookupTransform(robot_odom_frame_id_, odom_child_frame_id_, ros::Time(0), ros::Duration(0.1));
        Eigen::Matrix4f frame2odom = tf2::transformToEigen(frame_wrt_odom).cast<float>().matrix();

        geometry_msgs::TransformStamped map_wrt_odom;
        tf2::doTransform(map_wrt_frame, map_wrt_odom, frame_wrt_odom);

        tf2::Transform odom_wrt_map;
        tf2::fromMsg(map_wrt_odom.transform, odom_wrt_map);
        odom_wrt_map = odom_wrt_map.inverse();

        geometry_msgs::TransformStamped odom_trans;
        odom_trans.transform = tf2::toMsg(odom_wrt_map);
        odom_trans.header.stamp = stamp;
        odom_trans.header.frame_id = "map";
        odom_trans.child_frame_id = robot_odom_frame_id_;

        tf_broadcaster_.sendTransform(odom_trans);
      }
      else
      {
        geometry_msgs::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
        odom_trans.header.stamp = stamp;
        odom_trans.header.frame_id = "map";
        odom_trans.child_frame_id = odom_child_frame_id_;
        tf_broadcaster_.sendTransform(odom_trans);
      }
    }

    // publish the transform
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "map";

    tf::poseEigenToMsg(Eigen::Isometry3d(pose.cast<double>()), odom.pose.pose);
    odom.pose.covariance[0] = private_nh_.param<double>("cov_scaling_factor_x", 1.0) * fitness_score;
    odom.pose.covariance[7] = private_nh_.param<double>("cov_scaling_factor_y", 1.0) * fitness_score;
    odom.pose.covariance[14] = private_nh_.param<double>("cov_scaling_factor_z", 1.0) * fitness_score;
    odom.pose.covariance[21] = private_nh_.param<double>("cov_scaling_factor_R", 1.0) * fitness_score;
    odom.pose.covariance[28] = private_nh_.param<double>("cov_scaling_factor_P", 1.0) * fitness_score;
    odom.pose.covariance[35] = private_nh_.param<double>("cov_scaling_factor_Y", 1.0) * fitness_score;

    odom.child_frame_id = odom_child_frame_id_;
    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;

    pose_pub_.publish(odom);
  }

  /**
   * @brief publish scan matching status information
   */
  void publishScanMatchingStatus(const std_msgs::Header& header, pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned)
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
          tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator_->woPredictionError().get().cast<double>()))
              .transform);
    }

    if (pose_estimator_->imuPredictionError())
    {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = use_imu_ ? "imu" : "motion_model";
      status.prediction_errors.push_back(
          tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator_->imuPredictionError().get().cast<double>()))
              .transform);
    }

    if (pose_estimator_->odomPredictionError())
    {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = "odom";
      status.prediction_errors.push_back(
          tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator_->odomPredictionError().get().cast<double>()))
              .transform);
    }

    status_pub_.publish(status);
  }

private:
  // ROS
  ros::NodeHandle nh_;
  ros::NodeHandle mt_nh_;
  ros::NodeHandle private_nh_;

  bool use_odom_;
  ros::Time odom_stamp_last_;
  std::string robot_odom_frame_id_;
  std::string odom_child_frame_id_;
  bool enable_tf_;

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
  pcl::Registration<PointT, PointT>::Ptr registration_;

  // pose estimator
  std::mutex pose_estimator_mutex_;
  std::unique_ptr<hdl_localization::PoseEstimator> pose_estimator_;

  // global localization
  bool use_global_localization_;
  std::atomic_bool relocalizing_;
  std::unique_ptr<DeltaEstimater> delta_estimater_;

  pcl::PointCloud<PointT>::ConstPtr last_scan_;
  ros::ServiceServer relocalize_server_;
  ros::ServiceClient set_global_map_service_;
  ros::ServiceClient query_global_localization_service_;
};
}  // namespace hdl_localization

PLUGINLIB_EXPORT_CLASS(hdl_localization::HdlLocalizationNodelet, nodelet::Nodelet)
