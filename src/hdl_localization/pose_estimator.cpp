#include <hdl_localization/pose_estimator.hpp>

namespace hdl_localization
{
/**
 * @brief constructor
 * @param registration        registration method
 * @param pos                 initial position
 * @param quat                initial orientation
 * @param cool_time_duration  during "cool time", prediction is not performed
 */
PoseEstimator::PoseEstimator(pcl::Registration<PointT, PointT>::Ptr& registration,
                             const Eigen::Vector3f& initial_position, const Eigen::Quaternionf& initial_orientation,
                             double cool_time_duration)
  : registration_(registration), cool_time_duration_(cool_time_duration)
{
  last_observation_ = Eigen::Matrix4f::Identity();
  last_observation_.block<3, 3>(0, 0) = initial_orientation.toRotationMatrix();
  last_observation_.block<3, 1>(0, 3) = initial_position;

  process_noise_ = Eigen::MatrixXf::Identity(16, 16);
  process_noise_.middleRows(0, 3) *= 1.0;
  process_noise_.middleRows(3, 3) *= 1.0;
  process_noise_.middleRows(6, 4) *= 0.5;
  process_noise_.middleRows(10, 3) *= 1e-6;
  process_noise_.middleRows(13, 3) *= 1e-6;

  Eigen::MatrixXf measurement_noise = Eigen::MatrixXf::Identity(7, 7);
  measurement_noise.middleRows(0, 3) *= 0.01;
  measurement_noise.middleRows(3, 4) *= 0.001;

  Eigen::VectorXf mean(16);
  mean.middleRows(0, 3) = initial_position;
  mean.middleRows(3, 3).setZero();
  mean.middleRows(6, 4) = Eigen::Vector4f(initial_orientation.w(), initial_orientation.x(), initial_orientation.y(),
                                          initial_orientation.z())
                              .normalized();
  mean.middleRows(10, 3).setZero();
  mean.middleRows(13, 3).setZero();

  Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(16, 16) * 0.01;

  PoseSystem system;
  ukf_.reset(new kkl::alg::UnscentedKalmanFilterX<float, PoseSystem>(system, 16, 7, process_noise_, measurement_noise,
                                                                     mean, cov));
  // TODO: Change odom covariance constants to ROS params
  // or subscribe an odometry topic and use it's covariance
  odom_process_noise_ = Eigen::MatrixXf::Identity(16, 16) * 1e-5;
  odom_process_noise_.middleRows(6, 4) *= 1e-2;
}

PoseEstimator::~PoseEstimator()
{
}

/**
 * @brief predict
 * @param stamp    timestamp
 * @param acc      acceleration
 * @param gyro     angular velocity
 */
void PoseEstimator::predict(const ros::Time& stamp)
{
  if (init_stamp_.is_zero())
  {
    init_stamp_ = stamp;
  }

  if ((stamp - init_stamp_).toSec() < cool_time_duration_ || prev_stamp_.is_zero() || prev_stamp_ == stamp)
  {
    prev_stamp_ = stamp;
    return;
  }

  double dt = (stamp - prev_stamp_).toSec();
  prev_stamp_ = stamp;

  ukf_->setProcessNoiseCov(process_noise_ * dt);
  ukf_->system_.dt_ = dt;

  ukf_->predict();
}

/**
 * @brief predict
 * @param stamp    timestamp
 * @param imu_acc      acceleration
 * @param imu_gyro     angular velocity
 */
void PoseEstimator::predictImu(const ros::Time& stamp, const Eigen::Vector3f& imu_acc, const Eigen::Vector3f& imu_gyro)
{
  if (init_stamp_.is_zero())
  {
    init_stamp_ = stamp;
  }

  if ((stamp - init_stamp_).toSec() < cool_time_duration_ || prev_stamp_.is_zero() || prev_stamp_ == stamp)
  {
    prev_stamp_ = stamp;
    return;
  }

  double dt = (stamp - prev_stamp_).toSec();
  prev_stamp_ = stamp;

  ukf_->setProcessNoiseCov(process_noise_ * dt);
  ukf_->system_.dt_ = dt;
  ukf_->predictImu(imu_acc, imu_gyro);
}

/**
 * @brief predict_odom
 * @param stamp    timestamp
 * @param odom_twist_linear   linear velocity
 * @param odom_twist_angular  angular velocity
 */
void PoseEstimator::predictOdom(const ros::Time& stamp, const Eigen::Vector3f& odom_twist_linear,
                                const Eigen::Vector3f& odom_twist_angular)
{
  if ((stamp - init_stamp_).toSec() < cool_time_duration_ || prev_stamp_.is_zero() || prev_stamp_ == stamp)
  {
    prev_stamp_ = stamp;
    return;
  }

  double dt = (stamp - prev_stamp_).toSec();
  prev_stamp_ = stamp;

  ukf_->setProcessNoiseCov(odom_process_noise_ * dt);
  ukf_->system_.dt_ = dt;

  ukf_->predictOdom(odom_twist_linear, odom_twist_angular);
}

/**
 * @brief correct
 * @param cloud   input cloud
 * @return cloud aligned to the globalmap
 */
pcl::PointCloud<PoseEstimator::PointT>::Ptr PoseEstimator::correct(const ros::Time& stamp,
                                                                   const pcl::PointCloud<PointT>::ConstPtr& cloud,
                                                                   double& fitness_score)
{
  if (init_stamp_.is_zero())
  {
    init_stamp_ = stamp;
  }

  last_correction_stamp_ = stamp;

  Eigen::Matrix4f no_guess = last_observation_;
  Eigen::Matrix4f init_guess = matrix();

  pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
  registration_->setInputSource(cloud);
  registration_->align(*aligned, init_guess);
  fitness_score = registration_->getFitnessScore();

  Eigen::Matrix4f trans = registration_->getFinalTransformation();
  Eigen::Vector3f p = trans.block<3, 1>(0, 3);
  Eigen::Quaternionf q(trans.block<3, 3>(0, 0));

  if (quat().coeffs().dot(q.coeffs()) < 0.0f)
  {
    q.coeffs() *= -1.0f;
  }

  Eigen::VectorXf observation(7);
  observation.middleRows(0, 3) = p;
  observation.middleRows(3, 4) = Eigen::Vector4f(q.w(), q.x(), q.y(), q.z());
  last_observation_ = trans;

  wo_pred_error_ = no_guess.inverse() * registration_->getFinalTransformation();
  ukf_->correct(observation);
  motion_pred_error_ = init_guess.inverse() * registration_->getFinalTransformation();

  return aligned;
}

/* getters */
ros::Time PoseEstimator::lastCorrectionTime() const
{
  return last_correction_stamp_;
}

Eigen::Vector3f PoseEstimator::pos() const
{
  return Eigen::Vector3f(ukf_->mean_[0], ukf_->mean_[1], ukf_->mean_[2]);
}

Eigen::Vector3f PoseEstimator::vel() const
{
  return Eigen::Vector3f(ukf_->mean_[3], ukf_->mean_[4], ukf_->mean_[5]);
}

Eigen::Quaternionf PoseEstimator::quat() const
{
  return Eigen::Quaternionf(ukf_->mean_[6], ukf_->mean_[7], ukf_->mean_[8], ukf_->mean_[9]).normalized();
}

Eigen::Matrix4f PoseEstimator::matrix() const
{
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = quat().toRotationMatrix();
  m.block<3, 1>(0, 3) = pos();
  return m;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::woPredictionError() const
{
  return wo_pred_error_;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::motionPredictionError() const
{
  return motion_pred_error_;
}
}  // namespace hdl_localization
