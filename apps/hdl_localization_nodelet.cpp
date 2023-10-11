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

namespace hdl_localization {

class HdlLocalizationNodelet : public nodelet::Nodelet {
public:
  using PointT = pcl::PointXYZI;

  HdlLocalizationNodelet() : tf_buffer(), tf_listener(tf_buffer) {
    init_pose.setZero();
    init_orientation.x() = 0.0;
    init_orientation.y() = 0.0;
    init_orientation.z() = 0.0;
    init_orientation.w() = 1.0;
  }
  virtual ~HdlLocalizationNodelet() {}

  void onInit() override {
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();
    odom_stamp_last = ros::Time::now();

    initialize_params();
    map_frame_id = private_nh.param<std::string>("map_frame_id", "map");
    odom_frame_id = private_nh.param<std::string>("odom_frame_id", "odom");
    base_frame_id = private_nh.param<std::string>("base_frame_id", "base_link");

    specify_init_pose = private_nh.param<bool>("specify_init_pose", false);
    use_odom = private_nh.param<bool>("enable_robot_odometry_prediction", false);
    if (specify_init_pose) {
      init_pose.x() = private_nh.param<double>("init_pos_x", 0.0);
      init_pose.y() = private_nh.param<double>("init_pos_y", 0.0);
      init_pose.z() = private_nh.param<double>("init_pos_z", 0.0);
      init_orientation.x() = private_nh.param<double>("init_ori_x", 0.0);
      init_orientation.y() = private_nh.param<double>("init_ori_y", 0.0);
      init_orientation.z() = private_nh.param<double>("init_ori_z", 0.0);
      init_orientation.w() = private_nh.param<double>("init_ori_w", 1.0);
    } else if (use_odom) {
      odom_ready = false;
      initialize_on_odom = private_nh.param<bool>("initialize_on_odom", true);
      tf_buffer.canTransform(map_frame_id, odom_frame_id, ros::Time(0), ros::Duration(1.0));
      if (initialize_on_odom) {
        if (tf_buffer.canTransform(map_frame_id, base_frame_id, ros::Time(0), ros::Duration(10.0))) {
          geometry_msgs::TransformStamped tf_map2base = tf_buffer.lookupTransform(map_frame_id, base_frame_id, ros::Time(0), ros::Duration(10.0));
          init_pose.x() = tf_map2base.transform.translation.x;
          init_pose.y() = tf_map2base.transform.translation.y;
          init_pose.z() = tf_map2base.transform.translation.z;
          init_orientation.x() = tf_map2base.transform.rotation.x;
          init_orientation.y() = tf_map2base.transform.rotation.y;
          init_orientation.z() = tf_map2base.transform.rotation.z;
          init_orientation.w() = tf_map2base.transform.rotation.w;
        } else {
          NODELET_ERROR("Lookup transform failed %s -> %s", map_frame_id.c_str(), base_frame_id.c_str());
          initialize_on_odom = false;
        }
      }
    }

    enable_tf = private_nh.param<bool>("enable_tf", true);
    use_imu = private_nh.param<bool>("use_imu", true);
    if (use_odom && use_imu) {
      NODELET_WARN("[HdlLocalizationNodelet] Both use_odom and use_imu enabled -> disabling use_imu");
      use_imu = false;
    }
    invert_acc = private_nh.param<bool>("invert_acc", false);
    invert_gyro = private_nh.param<bool>("invert_gyro", false);

    // global localization
    use_global_localization = private_nh.param<bool>("use_global_localization", true);
    if (use_global_localization) {
      NODELET_INFO_STREAM("wait for global localization services");
      ros::service::waitForService("/hdl_global_localization/set_global_map");
      ros::service::waitForService("/hdl_global_localization/query");

      set_global_map_service = nh.serviceClient<hdl_global_localization::SetGlobalMap>("/hdl_global_localization/set_global_map");
      query_global_localization_service = nh.serviceClient<hdl_global_localization::QueryGlobalLocalization>("/hdl_global_localization/query");

      relocalize_server = nh.advertiseService("/relocalize", &HdlLocalizationNodelet::relocalize, this);
    }
    // initialize pose estimator
    NODELET_INFO("initialize pose estimator with specified parameters!!");
    pose_estimator.reset(new hdl_localization::PoseEstimator(
      registration,
      init_pose,
      init_orientation,
      private_nh.param<double>("cool_time_duration", 0.5),
      private_nh.param<double>("fitness_reject", 30.0),
      private_nh.param<double>("fitness_reliable", 0.03),
      private_nh.param<double>("linear_correction_gain", 1.0),
      private_nh.param<double>("angular_correction_gain", 1.0),
      private_nh.param<double>("angular_correction_distance_reject", 1.0),
      private_nh.param<double>("angular_correction_distance_reliable", 0.001)));
    if (use_imu) {
      NODELET_INFO("enable imu-based prediction");
      imu_sub = mt_nh.subscribe("/gpsimu_driver/imu_data", 5, &HdlLocalizationNodelet::imu_callback, this);
    }
    points_sub = mt_nh.subscribe("/velodyne_points", 5, &HdlLocalizationNodelet::points_callback, this);
    globalmap_sub = nh.subscribe("/globalmap", 1, &HdlLocalizationNodelet::globalmap_callback, this);
    initialpose_sub = nh.subscribe("/initialpose", 5, &HdlLocalizationNodelet::initialpose_callback, this);

    pose_pub = nh.advertise<nav_msgs::Odometry>("/odom", 5, false);
    aligned_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_points", 5, false);
    status_pub = nh.advertise<ScanMatchingStatus>("/status", 5, false);
  }

private:
  pcl::Registration<PointT, PointT>::Ptr create_registration() const {
    std::string reg_method = private_nh.param<std::string>("reg_method", "NDT_OMP");
    std::string ndt_neighbor_search_method = private_nh.param<std::string>("ndt_neighbor_search_method", "DIRECT7");
    double ndt_neighbor_search_radius = private_nh.param<double>("ndt_neighbor_search_radius", 2.0);
    double ndt_resolution = private_nh.param<double>("ndt_resolution", 1.0);

    if (reg_method == "NDT_OMP") {
      NODELET_INFO("NDT_OMP is selected");
      pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
      ndt->setTransformationEpsilon(0.01);
      ndt->setResolution(ndt_resolution);
      if (ndt_neighbor_search_method == "DIRECT1") {
        NODELET_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        NODELET_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
      } else {
        if (ndt_neighbor_search_method == "KDTREE") {
          NODELET_INFO("search_method KDTREE is selected");
        } else {
          NODELET_WARN("invalid search method was given");
          NODELET_WARN("default method is selected (KDTREE)");
        }
        ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
      }
      return ndt;
    } else if (reg_method.find("NDT_CUDA") != std::string::npos) {
      NODELET_INFO("NDT_CUDA is selected");
      boost::shared_ptr<fast_gicp::NDTCuda<PointT, PointT>> ndt(new fast_gicp::NDTCuda<PointT, PointT>);
      ndt->setResolution(ndt_resolution);

      if (reg_method.find("D2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::D2D);
      } else if (reg_method.find("P2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
      }

      if (ndt_neighbor_search_method == "DIRECT1") {
        NODELET_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        NODELET_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
      } else if (ndt_neighbor_search_method == "DIRECT_RADIUS") {
        NODELET_INFO_STREAM("search_method DIRECT_RADIUS is selected : " << ndt_neighbor_search_radius);
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT_RADIUS, ndt_neighbor_search_radius);
      } else {
        NODELET_WARN("invalid search method was given");
      }
      return ndt;
    }

    NODELET_ERROR_STREAM("unknown registration method:" << reg_method);
    return nullptr;
  }

  void initialize_params() {
    // intialize scan matching method
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
    if (downsample_resolution > 0.0) {
      boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
      voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter = voxelgrid;
    }else{
      ROS_WARN("Input downsample_filter is disabled");
      downsample_filter.reset();
    }


    NODELET_INFO("create registration method for localization");
    registration = create_registration();

    // global localization
    NODELET_INFO("create registration method for fallback during relocalization");
    relocalizing = false;
    delta_estimater.reset(new DeltaEstimater(create_registration()));
  }

private:
  /**
   * @brief callback for imu data
   * @param imu_msg
   */
  void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
    std::lock_guard<std::mutex> lock(imu_data_mutex);
    imu_data.push_back(imu_msg);
  }

  /**
   * @brief callback for point cloud data
   * @param points_msg
   */
  void points_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    if (!globalmap) {
      NODELET_WARN_THROTTLE(10.0, "Waiting for globalmap");
      return;
    }

    const auto& stamp = points_msg->header.stamp;
    pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *pcl_cloud);

    if (pcl_cloud->empty()) {
      NODELET_ERROR("cloud is empty!!");
      return;
    }

    ros::Time last_correction_time = pose_estimator->last_correction_time();
    // Skip calculation if timestamp is wrong
    if (stamp < last_correction_time) {
      return;
    }
    // transform pointcloud into base_frame_id
    std::string tfError;
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if (this->tf_buffer.canTransform(base_frame_id, pcl_cloud->header.frame_id, stamp, ros::Duration(0.1), &tfError)) {
      if (!pcl_ros::transformPointCloud(base_frame_id, *pcl_cloud, *cloud, this->tf_buffer)) {
        NODELET_ERROR("point cloud cannot be transformed into target frame!!");
        return;
      }
    } else {
      NODELET_ERROR(tfError.c_str());
      return;
    }

    auto filtered = downsample(cloud);
    last_scan = filtered;

    if (relocalizing) {
      delta_estimater->add_frame(filtered);
    }

    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
    if (!pose_estimator) {
      NODELET_ERROR("waiting for initial pose input!!");
      return;
    }
    Eigen::Matrix4f before = pose_estimator->matrix();

    if (use_imu) {
      // PointClouds + IMU prediction
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      auto imu_iter = imu_data.begin();
      for (imu_iter; imu_iter != imu_data.end(); imu_iter++) {
        if (stamp < (*imu_iter)->header.stamp) {
          break;
        }
        const auto& acc = (*imu_iter)->linear_acceleration;
        const auto& gyro = (*imu_iter)->angular_velocity;
        double acc_sign = invert_acc ? -1.0 : 1.0;
        double gyro_sign = invert_gyro ? -1.0 : 1.0;
        pose_estimator->predict_imu((*imu_iter)->header.stamp, acc_sign * Eigen::Vector3f(acc.x, acc.y, acc.z), gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
      }
      imu_data.erase(imu_data.begin(), imu_iter);
    } else if (use_odom) {
      if (!odom_ready) {
        sleep(0.1);
        odom_ready = tf_buffer.canTransform(map_frame_id, odom_frame_id, ros::Time::now(), ros::Duration(10.0));
        if (!odom_ready) {
          NODELET_ERROR("Waiting for %s -> %s transform", base_frame_id.c_str(), odom_frame_id.c_str());
          return;
        }
      }
      // PointClouds + Oodometry prediction
      if (tf_buffer.canTransform(base_frame_id, odom_stamp_last, base_frame_id, ros::Time(0), odom_frame_id, ros::Duration(0.1))) {
        // Get the amount of odometry movement since the last calculation
        // Coordinate system where the front of the robot is x
        geometry_msgs::TransformStamped odom_delta = tf_buffer.lookupTransform(base_frame_id, odom_stamp_last, base_frame_id, ros::Time(0), odom_frame_id);
        // Get the latest base_frame_id to get the time
        geometry_msgs::TransformStamped odom_now = tf_buffer.lookupTransform(odom_frame_id, base_frame_id, ros::Time(0));
        ros::Time odom_stamp = odom_now.header.stamp;
        ros::Duration odom_time_diff = odom_stamp - odom_stamp_last;
        double odom_time_diff_sec = odom_time_diff.toSec();
        if (odom_delta.header.stamp.isZero() || odom_time_diff_sec <= 0) {
          NODELET_WARN_THROTTLE(10.0, "Wrong timestamp detected: odom_time_diff_sec = %f", odom_time_diff_sec);
        } else {
          Eigen::Vector3f odom_travel_linear(odom_delta.transform.translation.x, odom_delta.transform.translation.y, odom_delta.transform.translation.z);
          Eigen::Vector3f odom_twist_linear = odom_travel_linear / odom_time_diff_sec;
          tf::Quaternion odom_travel_angular(odom_delta.transform.rotation.x, odom_delta.transform.rotation.y, odom_delta.transform.rotation.z, odom_delta.transform.rotation.w);
          double roll, pitch, yaw;
          tf::Matrix3x3(odom_travel_angular).getRPY(roll, pitch, yaw);
          Eigen::Vector3f odom_twist_angular(roll / odom_time_diff_sec, pitch / odom_time_diff_sec, yaw / odom_time_diff_sec);
          pose_estimator->predict_odom(odom_stamp, odom_twist_linear, odom_twist_angular);
        }
        odom_stamp_last = odom_stamp;
      } else {
        if (tf_buffer.canTransform(odom_frame_id, base_frame_id, ros::Time(0), ros::Duration(0.1))) {
          NODELET_WARN_THROTTLE(10.0, "The last timestamp is wrong, skip localization");
          // Get the latest base_frame_id to get the time
          geometry_msgs::TransformStamped odom_now = tf_buffer.lookupTransform(odom_frame_id, base_frame_id, ros::Time(0));
          odom_stamp_last = odom_now.header.stamp;
        } else {
          NODELET_WARN_STREAM("Failed to look up transform between " << cloud->header.frame_id << " and " << odom_frame_id);
        }
      }
    } else {
      // PointClouds-only prediction
      pose_estimator->predict(stamp);
    }
    // Perform scan matching using the calculated position as the initial value
    double fitness_score;
    auto aligned = pose_estimator->correct(stamp, filtered, fitness_score);

    if (aligned_pub.getNumSubscribers()) {
      aligned->header.frame_id = map_frame_id;
      aligned->header.stamp = cloud->header.stamp;
      aligned_pub.publish(aligned);
    }

    if (status_pub.getNumSubscribers()) {
      publish_scan_matching_status(points_msg->header, aligned);
    }

    publish_odometry(points_msg->header.stamp, pose_estimator->matrix(), fitness_score);
  }

  /**
   * @brief callback for globalmap input
   * @param points_msg
   */
  void globalmap_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    NODELET_INFO("globalmap received!");
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *cloud);
    globalmap = cloud;

    registration->setInputTarget(globalmap);

    if (use_global_localization) {
      NODELET_INFO("set globalmap for global localization!");
      hdl_global_localization::SetGlobalMap srv;
      pcl::toROSMsg(*globalmap, srv.request.global_map);

      if (!set_global_map_service.call(srv)) {
        NODELET_INFO("failed to set global map");
      } else {
        NODELET_INFO("done");
      }
    }
  }

  /**
   * @brief perform global localization to relocalize the sensor position
   * @param
   */
  bool relocalize(std_srvs::EmptyRequest& req, std_srvs::EmptyResponse& res) {
    if (last_scan == nullptr) {
      NODELET_INFO_STREAM("no scan has been received");
      return false;
    }

    relocalizing = true;
    delta_estimater->reset();
    pcl::PointCloud<PointT>::ConstPtr scan = last_scan;

    hdl_global_localization::QueryGlobalLocalization srv;
    pcl::toROSMsg(*scan, srv.request.cloud);
    srv.request.max_num_candidates = 1;

    if (!query_global_localization_service.call(srv) || srv.response.poses.empty()) {
      relocalizing = false;
      NODELET_INFO_STREAM("global localization failed");
      return false;
    }

    const auto& result = srv.response.poses[0];

    NODELET_INFO_STREAM("--- Global localization result ---");
    NODELET_INFO_STREAM("Trans :" << result.position.x << " " << result.position.y << " " << result.position.z);
    NODELET_INFO_STREAM("Quat  :" << result.orientation.x << " " << result.orientation.y << " " << result.orientation.z << " " << result.orientation.w);
    NODELET_INFO_STREAM("Error :" << srv.response.errors[0]);
    NODELET_INFO_STREAM("Inlier:" << srv.response.inlier_fractions[0]);

    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.linear() = Eigen::Quaternionf(result.orientation.w, result.orientation.x, result.orientation.y, result.orientation.z).toRotationMatrix();
    pose.translation() = Eigen::Vector3f(result.position.x, result.position.y, result.position.z);
    pose = pose * delta_estimater->estimated_delta();

    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    pose_estimator.reset(new hdl_localization::PoseEstimator(
      registration,
      pose.translation(),
      Eigen::Quaternionf(pose.linear()),
      private_nh.param<double>("cool_time_duration", 0.5),
      private_nh.param<double>("fitness_reject", 30.0),
      private_nh.param<double>("fitness_reliable", 0.03),
      private_nh.param<double>("linear_correction_gain", 1.0),
      private_nh.param<double>("angular_correction_gain", 1.0),
      private_nh.param<double>("angular_correction_distance_reject", 1.0),
      private_nh.param<double>("angular_correction_distance_reliable", 0.001)));

    relocalizing = false;

    return true;
  }

  /**
   * @brief callback for initial pose input ("2D Pose Estimate" on rviz)
   * @param pose_msg
   */
  void initialpose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg) {
    NODELET_INFO("[hdl_localization] initial pose received");
    geometry_msgs::TransformStamped tf_map2global = tf_buffer.lookupTransform(map_frame_id, msg->header.frame_id, ros::Time(0), ros::Duration(10.0));
    tf2::Vector3 vec_global2robot(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    tf2::Quaternion q_global2robot(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    // Get vector from odom to robot
    tf2::Vector3 vec_map2global(tf_map2global.transform.translation.x, tf_map2global.transform.translation.y, tf_map2global.transform.translation.z);
    tf2::Quaternion q_map2global(tf_map2global.transform.rotation.x, tf_map2global.transform.rotation.y, tf_map2global.transform.rotation.z, tf_map2global.transform.rotation.w);
    tf2::Vector3 vec_map2robot = vec_map2global + (tf2::Matrix3x3(q_map2global) * vec_global2robot);
    tf2::Quaternion q_map2robot = q_map2global * q_global2robot;
    init_pose.x() = vec_map2robot.x();
    init_pose.y() = vec_map2robot.y();
    init_pose.z() = vec_map2robot.z();

    init_orientation.x() = q_map2robot.x();
    init_orientation.y() = q_map2robot.y();
    init_orientation.z() = q_map2robot.z();
    init_orientation.w() = q_map2robot.w();
    if (use_odom) {
      tf_buffer.clear();
      odom_ready = false;
    }
    if (use_imu) {
      imu_data.clear();
    }
    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    pose_estimator.reset(new hdl_localization::PoseEstimator(
      registration,
      init_pose,
      init_orientation,
      private_nh.param<double>("cool_time_duration", 0.5),
      private_nh.param<double>("fitness_reject", 30.0),
      private_nh.param<double>("fitness_reliable", 0.03),
      private_nh.param<double>("linear_correction_gain", 1.0),
      private_nh.param<double>("angular_correction_gain", 1.0),
      private_nh.param<double>("angular_correction_distance_reject", 1.0),
      private_nh.param<double>("angular_correction_distance_reliable", 0.001)));
  }

  /**
   * @brief downsampling
   * @param cloud   input cloud
   * @return downsampled cloud
   */
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if (!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  /**
   * @brief publish odometry
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */
  void publish_odometry(const ros::Time& stamp, const Eigen::Matrix4f& pose, const double fitness_score) {
    // broadcast the transform over tf
    if (enable_tf) {
      if (tf_buffer.canTransform(odom_frame_id, base_frame_id, ros::Time(0))) {
        geometry_msgs::TransformStamped map_wrt_frame = tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
        map_wrt_frame.header.stamp = stamp;
        map_wrt_frame.header.frame_id = base_frame_id;
        map_wrt_frame.child_frame_id = map_frame_id;

        geometry_msgs::TransformStamped frame_wrt_odom = tf_buffer.lookupTransform(odom_frame_id, base_frame_id, ros::Time(0), ros::Duration(0.1));
        Eigen::Matrix4f frame2odom = tf2::transformToEigen(frame_wrt_odom).cast<float>().matrix();

        geometry_msgs::TransformStamped map_wrt_odom;
        tf2::doTransform(map_wrt_frame, map_wrt_odom, frame_wrt_odom);

        tf2::Transform odom_wrt_map;
        tf2::fromMsg(map_wrt_odom.transform, odom_wrt_map);
        odom_wrt_map = odom_wrt_map.inverse();

        geometry_msgs::TransformStamped odom_trans;
        odom_trans.transform = tf2::toMsg(odom_wrt_map);
        odom_trans.header.stamp = stamp;
        odom_trans.header.frame_id = map_frame_id;
        odom_trans.child_frame_id = odom_frame_id;

        tf_broadcaster.sendTransform(odom_trans);
      } else {
        geometry_msgs::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
        odom_trans.header.stamp = stamp;
        odom_trans.header.frame_id = map_frame_id;
        odom_trans.child_frame_id = base_frame_id;
        tf_broadcaster.sendTransform(odom_trans);
      }
    }

    // publish the transform
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = map_frame_id;
    odom.child_frame_id = base_frame_id;

    tf::poseEigenToMsg(Eigen::Isometry3d(pose.cast<double>()), odom.pose.pose);
    odom.pose.covariance[0] = private_nh.param<double>("cov_scaling_factor_x", 1.0) * fitness_score;
    odom.pose.covariance[7] = private_nh.param<double>("cov_scaling_factor_y", 1.0) * fitness_score;
    odom.pose.covariance[14] = private_nh.param<double>("cov_scaling_factor_z", 1.0) * fitness_score;
    odom.pose.covariance[21] = private_nh.param<double>("cov_scaling_factor_R", 1.0) * fitness_score;
    odom.pose.covariance[28] = private_nh.param<double>("cov_scaling_factor_P", 1.0) * fitness_score;
    odom.pose.covariance[35] = private_nh.param<double>("cov_scaling_factor_Y", 1.0) * fitness_score;

    odom.twist.twist.linear.x = 0;
    odom.twist.twist.linear.y = 0;
    odom.twist.twist.angular.z = 0.0;

    pose_pub.publish(odom);
  }

  /**
   * @brief publish scan matching status information
   */
  void publish_scan_matching_status(const std_msgs::Header& header, pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned) {
    ScanMatchingStatus status;
    status.header = header;

    status.has_converged = registration->hasConverged();
    status.matching_error = 0.0;

    const double max_correspondence_dist = private_nh.param<double>("status_max_correspondence_dist", 0.5);
    const double max_valid_point_dist = private_nh.param<double>("status_max_valid_point_dist", 25.0);

    int num_inliers = 0;
    int num_valid_points = 0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_dists;
    for (int i = 0; i < aligned->size(); i++) {
      const auto& pt = aligned->at(i);
      if (pt.getVector3fMap().norm() > max_valid_point_dist) {
        continue;
      }
      num_valid_points++;

      registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      if (k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
        status.matching_error += k_sq_dists[0];
        num_inliers++;
      }
    }

    status.matching_error /= num_inliers;
    status.inlier_fraction = static_cast<float>(num_inliers) / std::max(1, num_valid_points);
    status.relative_pose = tf2::eigenToTransform(Eigen::Isometry3d(registration->getFinalTransformation().cast<double>())).transform;

    status.prediction_labels.reserve(2);
    status.prediction_errors.reserve(2);

    std::vector<double> errors(6, 0.0);

    if (pose_estimator->without_pred_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = "without_prediction";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->without_pred_error().get().cast<double>())).transform);
    }

    if (pose_estimator->motion_pred_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      if(use_imu){
        status.prediction_labels.back().data = "imu_prediction";
      }else if(use_odom){
        status.prediction_labels.back().data = "odom_prediction";
      }else{
        status.prediction_labels.back().data = "motion_model";
      }
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->motion_pred_error().get().cast<double>())).transform);
    }

    status_pub.publish(status);
  }

private:
  // ROS
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;

  bool use_odom;
  bool odom_ready;
  bool initialize_on_odom;
  bool specify_init_pose;
  Eigen::Vector3f init_pose;
  Eigen::Quaternionf init_orientation;
  ros::Time odom_stamp_last;
  std::string odom_frame_id;
  std::string base_frame_id;
  std::string map_frame_id;
  bool enable_tf;

  bool use_imu;
  bool invert_acc;
  bool invert_gyro;
  ros::Subscriber imu_sub;
  ros::Subscriber points_sub;
  ros::Subscriber globalmap_sub;
  ros::Subscriber initialpose_sub;

  ros::Publisher pose_pub;
  ros::Publisher aligned_pub;
  ros::Publisher status_pub;

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener;
  tf2_ros::TransformBroadcaster tf_broadcaster;

  // imu input buffer
  std::mutex imu_data_mutex;
  std::vector<sensor_msgs::ImuConstPtr> imu_data;

  // globalmap and registration method
  pcl::PointCloud<PointT>::Ptr globalmap;
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration;

  // pose estimator
  std::mutex pose_estimator_mutex;
  std::unique_ptr<hdl_localization::PoseEstimator> pose_estimator;

  // global localization
  bool use_global_localization;
  std::atomic_bool relocalizing;
  std::unique_ptr<DeltaEstimater> delta_estimater;

  pcl::PointCloud<PointT>::ConstPtr last_scan;
  ros::ServiceServer relocalize_server;
  ros::ServiceClient set_global_map_service;
  ros::ServiceClient query_global_localization_service;
};
}  // namespace hdl_localization

PLUGINLIB_EXPORT_CLASS(hdl_localization::HdlLocalizationNodelet, nodelet::Nodelet)
