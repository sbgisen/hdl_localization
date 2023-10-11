#ifndef POSE_ESTIMATOR_HPP
#define POSE_ESTIMATOR_HPP

#include <memory>
#include <boost/optional.hpp>

#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

namespace kkl {
namespace alg {
template <typename T, class System>
class UnscentedKalmanFilterX;
}
}  // namespace kkl

namespace hdl_localization {

class PoseSystem;
class OdomSystem;

/**
 * @brief scan matching-based pose estimator
 */
class PoseEstimator {
public:
  using PointT = pcl::PointXYZI;

  /**
   * @brief constructor
   * @param registration        registration method
   * @param pos                 initial position
   * @param quat                initial orientation
   * @param cool_time_duration  during "cool time", prediction is not performed
   * @param fitness_reject     Do not process localization when scan matching fitness score is low
   */
  PoseEstimator(
    pcl::Registration<PointT, PointT>::Ptr& registration,
    const Eigen::Vector3f& pos,
    const Eigen::Quaternionf& quat,
    double cool_time_duration = 1.0,
    double fitness_reject = 100.0,
    double fitness_reliable = 0.1,
    double linear_correction_gain = 1.0,
    double angular_correction_gain = 1.0,
    double angular_correction_distance_reject = 1.0,
    double angular_correction_distance_reliable = 0.001);
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
  void predict_imu(const ros::Time& stamp, const Eigen::Vector3f& imu_acc, const Eigen::Vector3f& imu_gyro);

  /**
   * @brief update the state of the odomety-based pose estimation
   * @param stamp    timestamp
   * @param odom_twist_linear linear velocity
   * @param odom_twist_angular angular velocity
   */
  void predict_odom(const ros::Time& stamp, const Eigen::Vector3f& odom_twist_linear, const Eigen::Vector3f& odom_twist_angular);

  /**
   * @brief correct
   * @param cloud   input cloud
   * @return cloud aligned to the globalmap
   */
  pcl::PointCloud<PointT>::Ptr correct(const ros::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud, double& fitness_score);

  /* getters */
  ros::Time last_correction_time() const;

  Eigen::Vector3f pos() const;
  Eigen::Vector3f vel() const;
  Eigen::Quaternionf quat() const;
  Eigen::Matrix4f matrix() const;

  const boost::optional<Eigen::Matrix4f>& without_pred_error() const;
  const boost::optional<Eigen::Matrix4f>& motion_pred_error() const;

private:
  ros::Time init_stamp;             // when the estimator was initialized
  ros::Time prev_stamp;             // when the estimator was updated last time
  ros::Time last_correction_stamp;  // when the estimator performed the correction step
  double cool_time_duration;        // during "cool time", prediction is not performed
  double linear_correction_gain;
  double angular_correction_gain;
  double fitness_reject;  // Do not process localization when scan matching fitness score is low
  double fitness_reliable;
  double angular_correction_distance_reject;
  double angular_correction_distance_reliable;

  Eigen::MatrixXf process_noise;
  Eigen::MatrixXf odom_process_noise, imu_process_noise;
  std::unique_ptr<kkl::alg::UnscentedKalmanFilterX<float, PoseSystem>> ukf;

  Eigen::Matrix4f last_observation;
  boost::optional<Eigen::Matrix4f> without_pred_error_;
  boost::optional<Eigen::Matrix4f> motion_pred_error_;

  pcl::Registration<PointT, PointT>::Ptr registration;
};

}  // namespace hdl_localization

#endif  // POSE_ESTIMATOR_HPP
