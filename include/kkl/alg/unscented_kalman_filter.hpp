/**
 * UnscentedKalmanFilterX.hpp
 * @author koide
 * 16/02/01
 **/
#ifndef KKL_UNSCENTED_KALMAN_FILTER_X_HPP
#define KKL_UNSCENTED_KALMAN_FILTER_X_HPP

#include <random>
#include <Eigen/Dense>

namespace kkl {
namespace alg {

/**
 * @brief Unscented Kalman Filter class
 * @param T        scalar type
 * @param System   system class to be estimated
 */
template <typename T, class System>
class UnscentedKalmanFilterX {
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;

public:
  /**
   * @brief constructor
   * @param system               system to be estimated
   * @param state_dim            state vector dimension
   * @param measurement_dim      measurement vector dimension
   * @param process_noise        process noise covariance (state_dim x state_dim)
   * @param measurement_noise    measurement noise covariance (measurement_dim x measurement_dim)
   * @param init_mean            initial mean
   * @param init_cov             initial covariance
   */
  UnscentedKalmanFilterX(
    const System& system,
    int state_dim,
    int measurement_dim,
    const MatrixXt& process_noise,
    const MatrixXt& measurement_noise,
    const VectorXt& init_mean,
    const MatrixXt& init_cov)
  : state_dim_(state_dim),
    measurement_dim_(measurement_dim),
    sigma_points_samples_(2 * state_dim + 1),
    mean_(init_mean),
    cov_(init_cov),
    system_(system),
    process_noise_(process_noise),
    measurement_noise_(measurement_noise),
    lambda_(1),
    normal_dist_(0.0, 1.0) {
    weights_.resize(sigma_points_samples_, 1);
    sigma_points_.resize(sigma_points_samples_, state_dim);
    ext_weights_.resize(2 * (state_dim + measurement_dim) + 1, 1);
    ext_sigma_points_.resize(2 * (state_dim + measurement_dim) + 1, state_dim + measurement_dim);
    expected_measurements_.resize(2 * (state_dim + measurement_dim) + 1, measurement_dim);

    // initialize weights for unscented filter
    weights_[0] = lambda_ / (state_dim + lambda_);
    for (int i = 1; i < 2 * state_dim + 1; i++) {
      weights_[i] = 1 / (2 * (state_dim + lambda_));
    }

    // weights for extended state space which includes error variances
    ext_weights_[0] = lambda_ / (state_dim + measurement_dim + lambda_);
    for (int i = 1; i < 2 * (state_dim + measurement_dim) + 1; i++) {
      ext_weights_[i] = 1 / (2 * (state_dim + measurement_dim + lambda_));
    }
  }

  /**
   * @brief predict
   * @note state = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, 0, 0, 0, 0, 0, 0]
   */
  void predict() {
    // calculate sigma points
    ensurePositiveFinite(cov_);
    computeSigmaPoints(mean_, cov_, sigma_points_);

    for (int i = 0; i < sigma_points_samples_; i++) {
      sigma_points_.row(i) = system_.computeNextState(sigma_points_.row(i));
    }

    const auto& r = process_noise_;

    // unscented transform
    VectorXt mean_pred(mean_.size());
    MatrixXt cov_pred(cov_.rows(), cov_.cols());

    mean_pred.setZero();
    cov_pred.setZero();

    for (int i = 0; i < sigma_points_samples_; i++) {
      mean_pred += weights_[i] * sigma_points_.row(i);
    }

    for (int i = 0; i < sigma_points_samples_; i++) {
      VectorXt diff = sigma_points_.row(i).transpose() - mean_pred;
      cov_pred += weights_[i] * diff * diff.transpose();
    }

    cov_pred += r;

    mean_ = mean_pred;
    cov_ = cov_pred;
  }

  /**
   * @brief predict
   * @param imu_acc      acceleration
   * @param imu_gyro     angular velocity
   * @note state = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, acc_bias_x, acc_bias_y, acc_bias_z, gyro_bias_x, gyro_bias_y, gyro_bias_z]
   */
  void predictImu(const Eigen::Vector3f& imu_acc, const Eigen::Vector3f& imu_gyro) {
    // calculate sigma points
    ensurePositiveFinite(cov_);
    computeSigmaPoints(mean_, cov_, sigma_points_);

    for (int i = 0; i < sigma_points_samples_; i++) {
      sigma_points_.row(i) = system_.computeNextStateWithIMU(sigma_points_.row(i), imu_acc, imu_gyro);
    }

    const auto& r = process_noise_;

    // unscented transform
    VectorXt mean_pred(mean_.size());
    MatrixXt cov_pred(cov_.rows(), cov_.cols());

    mean_pred.setZero();
    cov_pred.setZero();

    for (int i = 0; i < sigma_points_samples_; i++) {
      mean_pred += weights_[i] * sigma_points_.row(i);
    }

    for (int i = 0; i < sigma_points_samples_; i++) {
      VectorXt diff = sigma_points_.row(i).transpose() - mean_pred;
      cov_pred += weights_[i] * diff * diff.transpose();
    }

    cov_pred += r;

    mean_ = mean_pred;
    cov_ = cov_pred;
  }

  /**
   * @brief predict
   * @param odom_twist_linear Velocity with x axis in front of the robot
   * @param odom_twist_angular angular velocity
   * @note state = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, 0, 0, 0, 0, 0, 0]
   */
  void predictOdom(const Eigen::Vector3f& odom_twist_linear, const Eigen::Vector3f& odom_twist_angular) {
    // calculate sigma points
    ensurePositiveFinite(cov_);
    computeSigmaPoints(mean_, cov_, sigma_points_);

    for (int i = 0; i < sigma_points_samples_; i++) {
      sigma_points_.row(i) = system_.computeNextStateWithOdom(sigma_points_.row(i), odom_twist_linear, odom_twist_angular);
    }

    const auto& r = process_noise_;

    // unscented transform
    VectorXt mean_pred(mean_.size());
    MatrixXt cov_pred(cov_.rows(), cov_.cols());

    mean_pred.setZero();
    cov_pred.setZero();

    for (int i = 0; i < sigma_points_samples_; i++) {
      mean_pred += weights_[i] * sigma_points_.row(i);
    }

    for (int i = 0; i < sigma_points_samples_; i++) {
      VectorXt diff = sigma_points_.row(i).transpose() - mean_pred;
      cov_pred += weights_[i] * diff * diff.transpose();
    }

    cov_pred += r;

    mean_ = mean_pred;
    cov_ = cov_pred;
  }

  /**
   * @brief correct
   * @param measurement  measurement vector
   */
  void correct(const VectorXt& measurement) {
    // create extended state space which includes error variances
    VectorXt ext_mean_pred = VectorXt::Zero(state_dim_ + measurement_dim_, 1);
    MatrixXt ext_cov_pred = MatrixXt::Zero(state_dim_ + measurement_dim_, state_dim_ + measurement_dim_);
    ext_mean_pred.topLeftCorner(state_dim_, 1) = VectorXt(mean_);
    ext_cov_pred.topLeftCorner(state_dim_, state_dim_) = MatrixXt(cov_);
    ext_cov_pred.bottomRightCorner(measurement_dim_, measurement_dim_) = measurement_noise_;

    ensurePositiveFinite(ext_cov_pred);
    computeSigmaPoints(ext_mean_pred, ext_cov_pred, ext_sigma_points_);

    // unscented transform
    expected_measurements_.setZero();
    for (int i = 0; i < ext_sigma_points_.rows(); i++) {
      expected_measurements_.row(i) = system_.computeObservation(ext_sigma_points_.row(i).transpose().topLeftCorner(state_dim_, 1));
      expected_measurements_.row(i) += VectorXt(ext_sigma_points_.row(i).transpose().bottomRightCorner(measurement_dim_, 1));
    }

    VectorXt expected_measurement_mean = VectorXt::Zero(measurement_dim_);
    for (int i = 0; i < ext_sigma_points_.rows(); i++) {
      expected_measurement_mean += ext_weights_[i] * expected_measurements_.row(i);
    }

    MatrixXt expected_measurement_cov = MatrixXt::Zero(measurement_dim_, measurement_dim_);
    for (int i = 0; i < ext_sigma_points_.rows(); i++) {
      VectorXt diff = expected_measurements_.row(i).transpose() - expected_measurement_mean;
      expected_measurement_cov += ext_weights_[i] * diff * diff.transpose();
    }

    // calculate transformed covariance
    MatrixXt sigma = MatrixXt::Zero(state_dim_ + measurement_dim_, measurement_dim_);
    for (int i = 0; i < ext_sigma_points_.rows(); i++) {
      auto diff_a = (ext_sigma_points_.row(i).transpose() - ext_mean_pred);
      auto diff_b = (expected_measurements_.row(i).transpose() - expected_measurement_mean);
      sigma += ext_weights_[i] * (diff_a * diff_b.transpose());
    }

    MatrixXt expected_inv = expected_measurement_cov.completeOrthogonalDecomposition().pseudoInverse();
    kalman_gain_ = sigma * expected_inv;
    const auto& k = kalman_gain_;

    VectorXt ext_mean = ext_mean_pred + k * (measurement - expected_measurement_mean);
    MatrixXt ext_cov = ext_cov_pred - k * expected_measurement_cov * k.transpose();

    mean_ = ext_mean.topLeftCorner(state_dim_, 1);
    cov_ = ext_cov.topLeftCorner(state_dim_, state_dim_);
  }

  /* getter */
  const VectorXt& getMean() const { return mean_; }
  const MatrixXt& getCov() const { return cov_; }
  const MatrixXt& getSigmaPoints() const { return sigma_points_; }

  System& getSystem() { return system_; }
  const System& getSystem() const { return system_; }
  const MatrixXt& getProcessNoiseCov() const { return process_noise_; }
  const MatrixXt& getMeasurementNoiseCov() const { return measurement_noise_; }
  const MatrixXt& getKalmanGain() const { return kalman_gain_; }

  /* setter */
  UnscentedKalmanFilterX& setMean(const VectorXt& m) {
    mean_ = m;
    return *this;
  }
  UnscentedKalmanFilterX& setCov(const MatrixXt& s) {
    cov_ = s;
    return *this;
  }

  UnscentedKalmanFilterX& setProcessNoiseCov(const MatrixXt& p) {
    process_noise_ = p;
    return *this;
  }
  UnscentedKalmanFilterX& setMeasurementNoiseCov(const MatrixXt& m) {
    measurement_noise_ = m;
    return *this;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  const int state_dim_;
  const int measurement_dim_;
  const int sigma_points_samples_;

public:
  VectorXt mean_;
  MatrixXt cov_;

  System system_;
  MatrixXt process_noise_;
  MatrixXt measurement_noise_;

  T lambda_;
  VectorXt weights_;

  MatrixXt sigma_points_;

  VectorXt ext_weights_;
  MatrixXt ext_sigma_points_;
  MatrixXt expected_measurements_;

  MatrixXt kalman_gain_;

  std::mt19937 mt_;
  std::normal_distribution<T> normal_dist_;

  /**
   * @brief compute sigma points
   * @param mean          mean
   * @param cov           covariance
   * @param sigma_points_  calculated sigma points
   */
  void computeSigmaPoints(const VectorXt& mean, const MatrixXt& cov, MatrixXt& sigma_points_) {
    const int n = mean.size();
    assert(cov.rows() == n && cov.cols() == n);

    Eigen::LLT<MatrixXt> llt;
    llt.compute((n + lambda_) * cov);
    MatrixXt l = llt.matrixL();

    sigma_points_.row(0) = mean;
    for (int i = 0; i < n; i++) {
      sigma_points_.row(1 + i * 2) = mean + l.col(i);
      sigma_points_.row(1 + i * 2 + 1) = mean - l.col(i);
    }
  }

  /**
   * @brief make covariance matrix positive finite
   * @param cov  covariance matrix
   */
  void ensurePositiveFinite(MatrixXt& cov) {
    return;
    const double eps = 1e-9;

    Eigen::EigenSolver<MatrixXt> solver(cov);
    MatrixXt d = solver.pseudoEigenvalueMatrix();
    MatrixXt v = solver.pseudoEigenvectors();
    for (int i = 0; i < d.rows(); i++) {
      if (d(i, i) < eps) {
        d(i, i) = eps;
      }
    }

    cov = v * d * v.inverse();
  }
};

}  // namespace alg
}  // namespace kkl

#endif