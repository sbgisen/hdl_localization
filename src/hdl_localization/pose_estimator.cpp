#include <hdl_localization/pose_estimator.hpp>
#include <pcl/filters/voxel_grid.h>
#include <hdl_localization/pose_system.hpp>
#include <kkl/alg/unscented_kalman_filter.hpp>

namespace hdl_localization {

/**
 * @brief Constructor
 * @param registration        Registration method
 * @param initial_position    Initial position
 * @param initial_orientation Initial orientation
 * @param cool_time_duration  Duration during which prediction is not performed ("cool time")
 * @param fitness_reject      Do not process localization when scan matching fitness score is low
 */
PoseEstimator::PoseEstimator(
  pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr& registration,
  const Eigen::Vector3f& initial_position,
  const Eigen::Quaternionf& initial_orientation,
  double cool_time_duration,
  double fitness_reject,
  double fitness_reliable,
  double linear_correction_gain,
  double angular_correction_gain,
  double angular_correction_distance_reject,
  double angular_correction_distance_reliable)
: registration_(registration),
  cool_time_duration_(cool_time_duration),
  fitness_reject_(fitness_reject),
  fitness_reliable_(fitness_reliable),
  linear_correction_gain_(linear_correction_gain),
  angular_correction_gain_(angular_correction_gain),
  angular_correction_distance_reject_(angular_correction_distance_reject),
  angular_correction_distance_reliable_(angular_correction_distance_reliable) {
  ROS_WARN(
    "Pamans: %f, %f, %f, %f, %f, %f, %f",
    cool_time_duration_,
    fitness_reject_,
    fitness_reliable_,
    linear_correction_gain_,
    angular_correction_gain_,
    angular_correction_distance_reject_,
    angular_correction_distance_reliable_);

  // Initialize initial pose
  Eigen::Matrix4f initial_pose = Eigen::Matrix4f::Identity();
  initial_pose.block<3, 1>(0, 3) = initial_position;
  initial_pose.block<3, 3>(0, 0) = initial_orientation.toRotationMatrix();
  last_observation_ = initial_pose;

  // Initialize process noise covariance matrix
  process_noise_ = Eigen::MatrixXf::Identity(16, 16);
  process_noise_.middleRows(0, 3) *= 1.0;    // Position
  process_noise_.middleRows(3, 3) *= 1.0;    // Velocity
  process_noise_.middleRows(6, 4) *= 1.0;    // Orientation
  process_noise_.middleRows(10, 3) *= 1e-6;  // Acceleration
  process_noise_.middleRows(13, 3) *= 1e-6;  // Angular velocity

  // Initialize measurement noise covariance matrix
  Eigen::MatrixXf measurement_noise = Eigen::MatrixXf::Identity(7, 7);
  measurement_noise.middleRows(0, 3) *= 1e-4;   // Position
  measurement_noise.middleRows(3, 4) *= 1e-5;  // Orientation

  // Initialize mean vector
  Eigen::VectorXf mean(16);
  mean.middleRows(0, 3) = initial_position;
  mean.middleRows(3, 3).setZero();
  mean.middleRows(6, 4) = Eigen::Vector4f(initial_orientation.w(), initial_orientation.x(), initial_orientation.y(), initial_orientation.z()).normalized();
  mean.middleRows(10, 3).setZero();
  mean.middleRows(13, 3).setZero();

  // TODO: Change odom covariance constants to ROS params
  // or subscribe an odometry topic and use its covariance
  // Initialize odometry process noise covariance matrix
  odom_process_noise_ = Eigen::MatrixXf::Identity(16, 16);
  odom_process_noise_.middleRows(0, 3) *= 1e-3;    // Position
  odom_process_noise_.middleRows(3, 3) *= 1e-9;    // Velocity
  odom_process_noise_.middleRows(6, 4) *= 1e-4;    // Orientation
  odom_process_noise_.middleRows(10, 3) *= 1e3;   // Acceleration
  odom_process_noise_.middleRows(13, 3) *= 1e-10;  // Angular velocity

  // Initialize IMU process noise covariance matrix
  imu_process_noise_ = Eigen::MatrixXf::Identity(16, 16);
  imu_process_noise_.middleRows(6, 4) *= 0.5;    // Orientation
  imu_process_noise_.middleRows(10, 3) *= 1e-6;  // Acceleration
  imu_process_noise_.middleRows(13, 3) *= 1e-6;  // Angular velocity

  // Initialize covariance matrix
  Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(16, 16) * 0.01;
  PoseEstimationSystem system;
  ukf_.reset(new kkl::alg::UnscentedKalmanFilterX<float, PoseEstimationSystem>(system, 16, 7, process_noise_, measurement_noise, mean, cov));
}

/**
 * @brief Destructor
 */
PoseEstimator::~PoseEstimator() {}

/**
 * @brief predict
 * @param stamp    timestamp
 * @param acc      acceleration
 * @param gyro     angular velocity
 */
void PoseEstimator::predict(const ros::Time& stamp) {
  if (init_stamp_.is_zero()) {
    init_stamp_ = stamp;
  }
  if ((stamp - init_stamp_).toSec() < cool_time_duration_ || prev_stamp_.is_zero() || prev_stamp_ == stamp) {
    prev_stamp_ = stamp;
    return;
  }
  double dt = (stamp - prev_stamp_).toSec();
  prev_stamp_ = stamp;
  ukf_->setProcessNoiseCov(process_noise_ * dt);
  ukf_->system_.time_step_ = dt;
  ukf_->predict();
}

/**
 * @brief predict using timestamp, IMU acceleration, and IMU angular velocity
 * @param stamp         Timestamp
 * @param imu_acc       IMU acceleration
 * @param imu_gyro      IMU angular velocity
 */
void PoseEstimator::predictImu(const ros::Time& stamp, const Eigen::Vector3f& imu_acc, const Eigen::Vector3f& imu_gyro) {
  if (init_stamp_.is_zero()) {
    init_stamp_ = stamp;
  }
  if ((stamp - init_stamp_).toSec() < cool_time_duration_ || prev_stamp_.is_zero() || prev_stamp_ == stamp) {
    prev_stamp_ = stamp;
    return;
  }
  double dt = (stamp - prev_stamp_).toSec();
  prev_stamp_ = stamp;
  ukf_->setProcessNoiseCov(imu_process_noise_ * dt);
  ukf_->system_.time_step_ = dt;
  ukf_->predictImu(imu_acc, imu_gyro);
}

/**
 * @brief predict using timestamp, odometry linear velocity, and odometry angular velocity
 * @param stamp             Timestamp
 * @param odom_twist_linear Odometry linear velocity
 * @param odom_twist_angular Odometry angular velocity
 */
void PoseEstimator::predictOdom(const ros::Time& stamp, const Eigen::Vector3f& odom_twist_linear, const Eigen::Vector3f& odom_twist_angular) {
  if (init_stamp_.is_zero()) {
    init_stamp_ = stamp;
  }
  if ((stamp - init_stamp_).toSec() < cool_time_duration_ || prev_stamp_.is_zero() || prev_stamp_ == stamp) {
    prev_stamp_ = stamp;
    return;
  }
  double dt = (stamp - prev_stamp_).toSec();
  prev_stamp_ = stamp;
  ukf_->setProcessNoiseCov(odom_process_noise_ * dt);
  ukf_->system_.time_step_ = dt;
  ukf_->predictOdom(odom_twist_linear, odom_twist_angular);
}

/**
 * @brief correct using point cloud
 * @param stamp         Timestamp
 * @param cloud         Input point cloud
 * @param fitness_score Fitness score of the scan matching
 * @return aligned point cloud
 */
pcl::PointCloud<PoseEstimator::PointT>::Ptr PoseEstimator::correct(const ros::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud, double pose_covariance[36]) {
  if (init_stamp_.is_zero()) {
    init_stamp_ = stamp;
  }
  last_correction_stamp_ = stamp;
  Eigen::Matrix4f no_guess = last_observation_;
  Eigen::Matrix4f init_guess = matrix();
  pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
  registration_->setInputSource(cloud);
  registration_->align(*aligned, init_guess);
  double fitness_score = registration_->getFitnessScore();
  if (fitness_score > fitness_reject_) {
    ROS_WARN_THROTTLE(5.0, "Scan matching fitness score is low (%f). Skip correction.", fitness_score);
    return aligned;
  }

  double max_probability = 3.3;
  double transform_probability = std::min(1.0, registration_->getTransformationProbability() / max_probability);
  if (transform_probability < 0.1) {
    ROS_WARN_THROTTLE(5.0, "Scan matching transformation probability is low (%f). Skip correction.", transform_probability);
    return aligned;
  }
  double iter = registration_->getFinalNumIteration();
  // ROS_WARN_THROTTLE(1.0, "Scan matching fitness score: %f (near: %f, prob: %f, iter: %f)", fitness_score, fitness_score_near, transform_probability, iter);

  double probability_scaling = transform_probability;
  double iter_scaling = std::max(std::min(1.0, iter / 30.0), 0.0);
  Eigen::Matrix4f trans = registration_->getFinalTransformation();
  Eigen::Vector3f p_measure = trans.block<3, 1>(0, 3);
  Eigen::Quaternionf q_measure(trans.block<3, 3>(0, 0));
  if (quat().coeffs().dot(q_measure.coeffs()) < 0.0f) {
    q_measure.coeffs() *= -1.0f;
  }
  // Get current estimation pose
  Eigen::Vector3f p_estimate = ukf_->mean_.head<3>();
  Eigen::Quaternionf q_estimate(ukf_->mean_[6], ukf_->mean_[7], ukf_->mean_[8], ukf_->mean_[9]);
  // If the value is unreliable, the correction is reduced
  double diff_linear_scaling = transform_probability;
  double diff_angular_scaling = transform_probability;
  // Correct only the position because the accuracy of angle correction is low when the position is significantly off.
  double diff_linear_norm = (p_measure - p_estimate).norm();
  double diff_angular_norm = fabs(q_estimate.angularDistance(q_measure));
  double fitness_scaling = fitness_reliable_ / (fitness_reliable_ + fitness_score);
  if (diff_linear_norm > angular_correction_distance_reject_) {
    diff_angular_scaling = 0.0;
  } else {
    diff_angular_scaling = angular_correction_distance_reliable_ / (angular_correction_distance_reliable_ + diff_linear_norm / angular_correction_distance_reject_);
  }
  // Limit max correction to prevent jumping
  double max_linear_correction = 3.0;
  double max_angular_correction = 1.0;
  if (diff_linear_norm > max_linear_correction) {
    diff_linear_scaling /= (diff_linear_norm / max_linear_correction);
    diff_angular_scaling /= (diff_linear_norm / max_linear_correction);
    diff_angular_norm /= (diff_linear_norm / max_linear_correction);
  }
  if (diff_angular_norm > max_angular_correction) {
    diff_linear_scaling /= (diff_angular_norm / max_angular_correction);
    diff_angular_scaling /= (diff_angular_norm / max_angular_correction);
    diff_linear_norm /= (diff_angular_norm / max_angular_correction);
  }
  // ROS_WARN_THROTTLE(1.0, "fitness_scaling: %f, probability_scaling: %f, iter_scaling: %f", fitness_scaling, probability_scaling, iter_scaling);
  // When fitness_score is large, the gain of correction is reduced
  diff_linear_scaling *= probability_scaling;
  diff_angular_scaling *= probability_scaling;
  diff_linear_scaling *=  linear_correction_gain_;
  diff_angular_scaling *=  angular_correction_gain_;
  diff_linear_scaling = std::max(std::min(diff_linear_scaling, 1.0), 0.0);
  diff_angular_scaling = std::max(std::min(diff_angular_scaling, 1.0), 0.0);
  // Add difference to current estimation
  Eigen::Vector3f p_measure_smooth = p_estimate + diff_linear_scaling * (p_measure - p_estimate);
  Eigen::Quaternionf q_measure_smooth = q_estimate.slerp(diff_angular_scaling, q_measure);
  // Update kalman filter
  Eigen::VectorXf observation(7);
  observation.middleRows(0, 3) = p_measure_smooth;
  observation.middleRows(3, 4) = Eigen::Vector4f(q_measure_smooth.w(), q_measure_smooth.x(), q_measure_smooth.y(), q_measure_smooth.z());
  last_observation_ = trans;
  // Fill data
  without_pred_error_ = no_guess.inverse() * trans;
  motion_pred_error_ = init_guess.inverse() * trans;
  ukf_->correct(observation);
  // Add remaining difference to covavriance
  Eigen::Vector3f linear_err = p_measure - p_measure_smooth;
  Eigen::Quaternionf q_err = q_measure * q_measure_smooth.inverse();
  transform_probability = std::max(transform_probability, 1e-6);
  double covariance = fitness_score + std::max(0.0, 1.0 / transform_probability - 1.0);
  for (int i = 0; i < 36; i++) {
    if (i % 7 == 0) {
      pose_covariance[i] = covariance;
    } else {
      pose_covariance[i] = 0;
    }
  }
  Eigen::Vector3f euler_error = Eigen::Vector3f(q_err.toRotationMatrix().eulerAngles(0, 1, 2));
  pose_covariance[0] += linear_err.x() * linear_err.x();
  pose_covariance[7] += linear_err.y() * linear_err.y();
  pose_covariance[35] += euler_error.z() * euler_error.z();
  return aligned;
}

/* getters */
/**
 * @brief Get the timestamp of the last correction
 * @return Timestamp of the last correction
 */
ros::Time PoseEstimator::lastCorrectionTime() const {
  return last_correction_stamp_;
}

/**
 * @brief Get the estimated position
 * @return Estimated position as a vector
 */
Eigen::Vector3f PoseEstimator::pos() const {
  return Eigen::Vector3f(ukf_->mean_[0], ukf_->mean_[1], ukf_->mean_[2]);
}

/**
 * @brief Get the estimated velocity
 * @return Estimated velocity as a vector
 */
Eigen::Vector3f PoseEstimator::vel() const {
  return Eigen::Vector3f(ukf_->mean_[3], ukf_->mean_[4], ukf_->mean_[5]);
}

/**
 * @brief Get the estimated orientation
 * @return Estimated orientation as a quaternion
 */
Eigen::Quaternionf PoseEstimator::quat() const {
  return Eigen::Quaternionf(ukf_->mean_[6], ukf_->mean_[7], ukf_->mean_[8], ukf_->mean_[9]).normalized();
}

/**
 * @brief Get the estimated transformation matrix
 * @return Estimated transformation matrix
 */
Eigen::Matrix4f PoseEstimator::matrix() const {
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = quat().toRotationMatrix();
  m.block<3, 1>(0, 3) = pos();
  return m;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::withoutPredError() const {
  return without_pred_error_;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::motionPredError() const {
  return motion_pred_error_;
}
}  // namespace hdl_localization
