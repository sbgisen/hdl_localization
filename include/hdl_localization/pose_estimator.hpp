#ifndef POSE_ESTIMATOR_HPP
#define POSE_ESTIMATOR_HPP

#include <memory>
#include <boost/optional.hpp>

#include <ros/ros.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pclomp/ndt_omp.h>

#include <hdl_localization/pose_system.hpp>
#include <ukf/unscented_kalman_filter.hpp>

namespace kkl
{
namespace alg
{
template <typename T, class System>
class UnscentedKalmanFilterX;
}
}  // namespace kkl

namespace hdl_localization
{
class PoseSystem;
class OdomSystem;

/**
 * @brief scan matching-based pose estimator
 */
class PoseEstimator
{
public:
  using PointT = pcl::PointXYZI;

  /**
   * @brief constructor
   * @param registration        registration method
   * @param pos                 initial position
   * @param quat                initial orientation
   * @param cool_time_duration  during "cool time", prediction is not performed
   */
  PoseEstimator(pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr& registration,
                const Eigen::Vector3f& initial_position, const Eigen::Quaternionf& initial_orientation,
                double cool_time_duration = 1.0);
  ~PoseEstimator();

  /**
   * @brief predict
   * @param stamp    timestamp
   */
  void predict(const ros::Time& stamp);

  /**
   * @brief update the state of the IMU-based pose estimation
   * @param stamp    timestamp
   * @param imu_acc      acceleration
   * @param imu_gyro     angular velocity
   */
  void predictImu(const ros::Time& stamp, const Eigen::Vector3f& imu_acc, const Eigen::Vector3f& imu_gyro);

  /**
   * @brief update the state of the odomety-based pose estimation
   * @param stamp    timestamp
   * @param odom_twist_linear linear velocity
   * @param odom_twist_angular angular velocity
   */
  void predictOdom(const ros::Time& stamp, const Eigen::Vector3f& odom_twist_linear,
                   const Eigen::Vector3f& odom_twist_angular);

  /**
   * @brief correct
   * @param cloud   input cloud
   * @return cloud aligned to the globalmap
   */
  pcl::PointCloud<PointT>::Ptr correct(const ros::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud,
                                       double& fitness_score);

  /* getters */
  ros::Time lastCorrectionTime() const;

  Eigen::Vector3f pos() const;
  Eigen::Vector3f vel() const;
  Eigen::Quaternionf quat() const;
  Eigen::Matrix4f matrix() const;

  const boost::optional<Eigen::Matrix4f>& woPredictionError() const;
  const boost::optional<Eigen::Matrix4f>& motionPredictionError() const;

private:
  ros::Time init_stamp_;             // when the estimator was initialized
  ros::Time prev_stamp_;             // when the estimator was updated last time
  ros::Time last_correction_stamp_;  // when the estimator performed the correction step
  double cool_time_duration_;        //

  Eigen::MatrixXf process_noise_, odom_process_noise_;
  std::unique_ptr<kkl::alg::UnscentedKalmanFilterX<float, PoseSystem>> ukf_;

  Eigen::Matrix4f last_observation_;
  boost::optional<Eigen::Matrix4f> wo_pred_error_;
  boost::optional<Eigen::Matrix4f> motion_pred_error_;

  pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr registration_;
};

}  // namespace hdl_localization

#endif  // POSE_ESTIMATOR_HPP
