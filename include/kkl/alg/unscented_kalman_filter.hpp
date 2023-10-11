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
 * @param T        scaler type
 * @param System   system class to be estimated
 */
template<typename T, class System>
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
   * @param measurement_noise    measurement noise covariance (measurement_dim x measuremend_dim)
   * @param mean                 initial mean
   * @param cov                  initial covariance
   */
  UnscentedKalmanFilterX(const System& system, int state_dim, int measurement_dim, const MatrixXt& process_noise, const MatrixXt& measurement_noise, const VectorXt& mean, const MatrixXt& cov)
    : state_dim(state_dim),
    measurement_dim(measurement_dim),
    sigma_points_samples(2 * state_dim + 1),
    mean(mean),
    cov(cov),
    system(system),
    process_noise(process_noise),
    measurement_noise(measurement_noise),
    lambda(1),
    normal_dist(0.0, 1.0)
  {
    weights.resize(sigma_points_samples, 1);
    sigma_points.resize(sigma_points_samples, state_dim);
    ext_weights.resize(2 * (state_dim + measurement_dim) + 1, 1);
    ext_sigma_points.resize(2 * (state_dim + measurement_dim) + 1, state_dim + measurement_dim);
    expected_measurements.resize(2 * (state_dim + measurement_dim) + 1, measurement_dim);
    // initialize weights for unscented filter
    weights[0] = lambda / (state_dim + lambda);
    for (int i = 1; i < 2 * state_dim + 1; i++) {
      weights[i] = 1 / (2 * (state_dim + lambda));
    }
    // weights for extended state space which includes error variances
    ext_weights[0] = lambda / (state_dim + measurement_dim + lambda);
    for (int i = 1; i < 2 * (state_dim + measurement_dim) + 1; i++) {
      ext_weights[i] = 1 / (2 * (state_dim + measurement_dim + lambda));
    }
  }

  /**
   * @brief predict
   * @note state = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, 0, 0, 0, 0, 0, 0]
   */
  void predict() {
    // calculate sigma points
    ensurePositiveFinite(cov);
    computeSigmaPoints(mean, cov, sigma_points);
    for (int i = 0; i < sigma_points_samples; i++) {
      sigma_points.row(i) = system.f(sigma_points.row(i));
    }

    const auto& R = process_noise;

    // unscented transform
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());

    mean_pred.setZero();
    cov_pred.setZero();
    for (int i = 0; i < sigma_points_samples; i++) {
      mean_pred += weights[i] * sigma_points.row(i);
    }
    for (int i = 0; i < sigma_points_samples; i++) {
      VectorXt diff = sigma_points.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();
    }
    cov_pred += R;
    mean = mean_pred;
    cov = cov_pred;
  }

  /**
   * @brief predict
   * @param imu_acc      acceleration
   * @param imu_gyro     angular velocity
   * @note state = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, acc_bias_x, acc_bias_y, acc_bias_z, gyro_bias_x, gyro_bias_y, gyro_bias_z]
   */
  void predict_imu(const Eigen::Vector3f& imu_acc, const Eigen::Vector3f& imu_gyro) {
    // calculate sigma points
    ensurePositiveFinite(cov);
    computeSigmaPoints(mean, cov, sigma_points);
    for (int i = 0; i < sigma_points_samples; i++) {
      sigma_points.row(i) = system.f_imu(sigma_points.row(i), imu_acc, imu_gyro);
    }

    const auto& R = process_noise;

    // unscented transform
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());

    mean_pred.setZero();
    cov_pred.setZero();
    for (int i = 0; i < sigma_points_samples; i++) {
      mean_pred += weights[i] * sigma_points.row(i);
    }
    for (int i = 0; i < sigma_points_samples; i++) {
      VectorXt diff = sigma_points.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();
    }
    cov_pred += R;
    mean = mean_pred;
    cov = cov_pred;
  }

  /**
   * @brief predict
   * @param odom_twist_linear Velocity with x axis in front of the robot
   * @param odom_twist_angular angular velocity
   * @note state = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, 0, 0, 0, 0, 0, 0]
   */
  void predict_odom(Eigen::Vector3f odom_twist_linear, Eigen::Vector3f odom_twist_angular) {
    // calculate sigma points
    ensurePositiveFinite(cov);
    computeSigmaPoints(mean, cov, sigma_points);
    for (int i = 0; i < sigma_points_samples; i++) {
      sigma_points.row(i) = system.f_odom(sigma_points.row(i), odom_twist_linear, odom_twist_angular);
    }
    const auto& R = process_noise;
    // unscented transform
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());
    mean_pred.setZero();
    cov_pred.setZero();
    for (int i = 0; i < sigma_points_samples; i++) {
      mean_pred += weights[i] * sigma_points.row(i);
    }
    for (int i = 0; i < sigma_points_samples; i++) {
      VectorXt diff = sigma_points.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();
    }
    cov_pred += R;
    mean = mean_pred;
    cov = cov_pred;
  }

  /**
   * @brief correct
   * @param measurement  measurement vector
   */
  void correct(const VectorXt& measurement) {
    // create extended state space which includes error variances
    VectorXt ext_mean_pred = VectorXt::Zero(state_dim + measurement_dim, 1);
    MatrixXt ext_cov_pred = MatrixXt::Zero(state_dim + measurement_dim, state_dim + measurement_dim);
    ext_mean_pred.topLeftCorner(state_dim, 1) = VectorXt(mean);
    ext_cov_pred.topLeftCorner(state_dim, state_dim) = MatrixXt(cov);
    ext_cov_pred.bottomRightCorner(measurement_dim, measurement_dim) = measurement_noise;

    ensurePositiveFinite(ext_cov_pred);
    computeSigmaPoints(ext_mean_pred, ext_cov_pred, ext_sigma_points);

    // unscented transform
    expected_measurements.setZero();
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurements.row(i) = system.h(ext_sigma_points.row(i).transpose().topLeftCorner(state_dim, 1));
      expected_measurements.row(i) += VectorXt(ext_sigma_points.row(i).transpose().bottomRightCorner(measurement_dim, 1));
    }
    VectorXt expected_measurement_mean = VectorXt::Zero(measurement_dim);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurement_mean += ext_weights[i] * expected_measurements.row(i);
    }
    MatrixXt expected_measurement_cov = MatrixXt::Zero(measurement_dim, measurement_dim);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      VectorXt diff = expected_measurements.row(i).transpose() - expected_measurement_mean;
      expected_measurement_cov += ext_weights[i] * diff * diff.transpose();
    }

    // calculated transformed covariance
    MatrixXt sigma = MatrixXt::Zero(state_dim + measurement_dim, measurement_dim);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      auto diffA = (ext_sigma_points.row(i).transpose() - ext_mean_pred);
      auto diffB = (expected_measurements.row(i).transpose() - expected_measurement_mean);
      sigma += ext_weights[i] * (diffA * diffB.transpose());
    }

    MatrixXt expexted_inv = expected_measurement_cov.completeOrthogonalDecomposition().pseudoInverse();
    kalman_gain = sigma * expexted_inv;
    const auto& K = kalman_gain;

    VectorXt ext_mean = ext_mean_pred + K * (measurement - expected_measurement_mean);
    MatrixXt ext_cov = ext_cov_pred - K * expected_measurement_cov * K.transpose();

    mean = ext_mean.topLeftCorner(state_dim, 1);
    cov = ext_cov.topLeftCorner(state_dim, state_dim);
  }

  /*			getter			*/
  const VectorXt& getMean() const { return mean; }
  const MatrixXt& getCov() const { return cov; }
  const MatrixXt& getSigmaPoints() const { return sigma_points; }

  System& getSystem() { return system; }
  const System& getSystem() const { return system; }
  const MatrixXt& getProcessNoiseCov() const { return process_noise; }
  const MatrixXt& getMeasurementNoiseCov() const { return measurement_noise; }

  const MatrixXt& getKalmanGain() const { return kalman_gain; }

  /*			setter			*/
  UnscentedKalmanFilterX& setMean(const VectorXt& m) { mean = m;			return *this; }
  UnscentedKalmanFilterX& setCov(const MatrixXt& s) { cov = s;			return *this; }

  UnscentedKalmanFilterX& setProcessNoiseCov(const MatrixXt& p) { process_noise = p;			return *this; }
  UnscentedKalmanFilterX& setMeasurementNoiseCov(const MatrixXt& m) { measurement_noise = m;	return *this; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  const int state_dim;
  const int measurement_dim;
  const int sigma_points_samples;

public:
  VectorXt mean;
  MatrixXt cov;

  System system;
  MatrixXt process_noise;		//
  MatrixXt measurement_noise;	//

  T lambda;
  VectorXt weights;

  MatrixXt sigma_points;

  VectorXt ext_weights;
  MatrixXt ext_sigma_points;
  MatrixXt expected_measurements;

private:
  /**
   * @brief compute sigma points
   * @param mean          mean
   * @param cov           covariance
   * @param sigma_points  calculated sigma points
   */
  void computeSigmaPoints(const VectorXt& mean, const MatrixXt& cov, MatrixXt& sigma_points) {
    const int n = mean.size();
    assert(cov.rows() == n && cov.cols() == n);

    Eigen::LLT<MatrixXt> llt;
    llt.compute((n + lambda) * cov);
    MatrixXt l = llt.matrixL();

    sigma_points.row(0) = mean;
    for (int i = 0; i < n; i++) {
      sigma_points.row(1 + i * 2) = mean + l.col(i);
      sigma_points.row(1 + i * 2 + 1) = mean - l.col(i);
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
    MatrixXt D = solver.pseudoEigenvalueMatrix();
    MatrixXt V = solver.pseudoEigenvectors();
    for (int i = 0; i < D.rows(); i++) {
      if (D(i, i) < eps) {
        D(i, i) = eps;
      }
    }

    cov = V * D * V.inverse();
  }

public:
  MatrixXt kalman_gain;

  std::mt19937 mt;
  std::normal_distribution<T> normal_dist;
};

  }
}


#endif
