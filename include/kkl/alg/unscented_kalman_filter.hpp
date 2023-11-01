/**
 * UnscentedKalmanFilterX.hpp
 * @author koide
 * 16/02/01
 **/
#ifndef KKL_UNSCENTED_KALMAN_FILTER_X_HPP
#define KKL_UNSCENTED_KALMAN_FILTER_X_HPP

#include <random>
#include <Eigen/Dense>

namespace kkl
{
namespace alg
{

/**
 * @brief Unscented Kalman Filter class
 * @param T        scaler type
 * @param System   system class to be estimated
 */
template <typename T, class System>
class UnscentedKalmanFilterX
{
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;

public:
  /**
   * @brief constructor
   * @param system               system to be estimated
   * @param state_dim            state vector dimension
   * @param input_dim            input vector dimension
   * @param measurement_dim      measurement vector dimension
   * @param process_noise        process noise covariance (state_dim x state_dim)
   * @param measurement_noise    measurement noise covariance (measurement_dim x measuremend_dim)
   * @param mean                 initial mean
   * @param cov                  initial covariance
   */
  UnscentedKalmanFilterX(const System& system, int state_dim, int input_dim, int measurement_dim,
                         const MatrixXt& process_noise, const MatrixXt& measurement_noise, const VectorXt& mean,
                         const MatrixXt& cov)
    : state_dim_(state_dim)
    , input_dim_(input_dim)
    , measurement_dim_(measurement_dim)
    , N_(state_dim)
    , M_(input_dim)
    , K_(measurement_dim)
    , S_(2 * state_dim + 1)
    , mean_(mean)
    , cov_(cov)
    , system_(system)
    , process_noise_(process_noise)
    , measurement_noise_(measurement_noise)
    , lambda_(1)
    , normal_dist_(0.0, 1.0)
  {
    weights_.resize(S_, 1);
    sigma_points_.resize(S_, N_);
    ext_weights_.resize(2 * (N_ + K_) + 1, 1);
    ext_sigma_points_.resize(2 * (N_ + K_) + 1, N_ + K_);
    expected_measurements_.resize(2 * (N_ + K_) + 1, K_);

    // initialize weights for unscented filter
    weights[0] = lambda / (N + lambda);
    for (int i = 1; i < 2 * N_ + 1; i++)
    {
      weights[i] = 1 / (2 * (N + lambda));
    }

    // weights for extended state space which includes error variances
    ext_weights[0] = lambda / (N + K + lambda);
    for (int i = 1; i < 2 * (N_ + K_) + 1; i++)
    {
      ext_weights[i] = 1 / (2 * (N + K + lambda));
    }
  }

  /**
   * @brief predict
   * @note state = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, 0, 0, 0, 0, 0, 0]
   */
  void predict()
  {
    // calculate sigma points
    ensurePositiveFinite(cov_);
    computeSigmaPoints(mean_, cov_, sigma_points_);
    for (int i = 0; i < S_; i++)
    {
      sigma_points_.row(i) = system_.f(sigma_points_.row(i));
    }

    const auto& r = process_noise_;

    // unscented transform
    VectorXt mean_pred(mean_.size());
    MatrixXt cov_pred(cov_.rows(), cov_.cols());

    mean_pred.setZero();
    cov_pred.setZero();
    for (int i = 0; i < S_; i++)
    {
      mean_pred += weights[i] * sigma_points.row(i);
    }
    for (int i = 0; i < S_; i++)
    {
      VectorXt diff = sigma_points_.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();
    }
    cov_pred += r;

    mean_ = mean_pred;
    cov_ = cov_pred;
  }

  /**
   * @brief predict
   * @param imu_acc      acceleration
   * @param imu_gyro     angular velocity
   * @note state = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, acc_bias_x, acc_bias_y, acc_bias_z, gyro_bias_x,
   * gyro_bias_y, gyro_bias_z]
   */
  void predictImu(const Eigen::Vector3f& imu_acc, const Eigen::Vector3f& imu_gyro)
  {
    // calculate sigma points
    ensurePositiveFinite(cov_);
    computeSigmaPoints(mean_, cov_, sigma_points_);
    for (int i = 0; i < S_; i++)
    {
      sigma_points_.row(i) = system_.f_imu(sigma_points_.row(i), imu_acc, imu_gyro);
    }

    const auto& r = process_noise_;

    // unscented transform
    VectorXt mean_pred(mean_.size());
    MatrixXt cov_pred(cov_.rows(), cov_.cols());

    mean_pred.setZero();
    cov_pred.setZero();
    for (int i = 0; i < S_; i++)
    {
      mean_pred += weights[i] * sigma_points.row(i);
    }
    for (int i = 0; i < S_; i++)
    {
      VectorXt diff = sigma_points_.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();
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
  void predictOdom(Eigen::Vector3f /*odom_twist_linear*/, Eigen::Vector3f /*odom_twist_angular*/)
  {
    // calculate sigma points
    ensurePositiveFinite(cov_);
    computeSigmaPoints(mean_, cov_, sigma_points_);
    for (int i = 0; i < S_; i++)
    {
      sigma_points.row(i) = system.f_odom(sigma_points.row(i), odom_twist_linear, odom_twist_angular);
    }
    const auto& r = process_noise_;
    // unscented transform
    VectorXt mean_pred(mean_.size());
    MatrixXt cov_pred(cov_.rows(), cov_.cols());
    mean_pred.setZero();
    cov_pred.setZero();
    for (int i = 0; i < S_; i++)
    {
      mean_pred += weights[i] * sigma_points.row(i);
    }
    for (int i = 0; i < S_; i++)
    {
      VectorXt diff = sigma_points_.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();
    }
    cov_pred += r;
    mean_ = mean_pred;
    cov_ = cov_pred;
  }

  /**
   * @brief correct
   * @param measurement  measurement vector
   */
  void correct(const VectorXt& measurement)
  {
    // create extended state space which includes error variances
    VectorXt ext_mean_pred = VectorXt::Zero(N + K, 1);
    MatrixXt ext_cov_pred = MatrixXt::Zero(N + K, N + K);
    ext_mean_pred.topLeftCorner(N_, 1) = VectorXt(mean_);
    ext_cov_pred.topLeftCorner(N_, N_) = MatrixXt(cov_);
    ext_cov_pred.bottomRightCorner(K_, K_) = measurement_noise_;

    ensurePositiveFinite(ext_cov_pred);
    computeSigmaPoints(ext_mean_pred, ext_cov_pred, ext_sigma_points_);

    // unscented transform
    expected_measurements_.setZero();
    for (int i = 0; i < ext_sigma_points_.rows(); i++)
    {
      expected_measurements_.row(i) = system_.h(ext_sigma_points_.row(i).transpose().topLeftCorner(N_, 1));
      expected_measurements_.row(i) += VectorXt(ext_sigma_points_.row(i).transpose().bottomRightCorner(K_, 1));
    }

    VectorXt expected_measurement_mean = VectorXt::Zero(K);
    for (int i = 0; i < ext_sigma_points_.rows(); i++)
    {
      expected_measurement_mean += ext_weights[i] * expected_measurements.row(i);
    }
    MatrixXt expected_measurement_cov = MatrixXt::Zero(K, K);
    for (int i = 0; i < ext_sigma_points_.rows(); i++)
    {
      VectorXt diff = expected_measurements_.row(i).transpose() - expected_measurement_mean;
      expected_measurement_cov += ext_weights[i] * diff * diff.transpose();
    }

    // calculated transformed covariance
    MatrixXt sigma = MatrixXt::Zero(N + K, K);
    for (int i = 0; i < ext_sigma_points_.rows(); i++)
    {
      auto diff_a = (ext_sigma_points_.row(i).transpose() - ext_mean_pred);
      auto diff_b = (expected_measurements_.row(i).transpose() - expected_measurement_mean);
      sigma += ext_weights[i] * (diffA * diffB.transpose());
    }

    MatrixXt expexted_inv = expected_measurement_cov.completeOrthogonalDecomposition().pseudoInverse();
    kalman_gain_ = sigma * expexted_inv;
    const auto& k = kalman_gain_;

    VectorXt ext_mean = ext_mean_pred + k * (measurement - expected_measurement_mean);
    MatrixXt ext_cov = ext_cov_pred - k * expected_measurement_cov * k.transpose();

    mean_ = ext_mean.topLeftCorner(N_, 1);
    cov_ = ext_cov.topLeftCorner(N_, N_);
  }

  /*			getter			*/
  const VectorXt& getMean() const
  {
    return mean_;
  }
  const MatrixXt& getCov() const
  {
    return cov_;
  }
  const MatrixXt& getSigmaPoints() const
  {
    return sigma_points_;
  }

  System& getSystem()
  {
    return system_;
  }
  const System& getSystem() const
  {
    return system_;
  }
  const MatrixXt& getProcessNoiseCov() const
  {
    return process_noise_;
  }
  const MatrixXt& getMeasurementNoiseCov() const
  {
    return measurement_noise_;
  }

  const MatrixXt& getKalmanGain() const
  {
    return kalman_gain_;
  }

  /*			setter			*/
  UnscentedKalmanFilterX& setMean(const VectorXt& m)
  {
    mean_ = m;
    return *this;
  }
  UnscentedKalmanFilterX& setCov(const MatrixXt& s)
  {
    cov_ = s;
    return *this;
  }

  UnscentedKalmanFilterX& setProcessNoiseCov(const MatrixXt& p)
  {
    process_noise_ = p;
    return *this;
  }
  UnscentedKalmanFilterX& setMeasurementNoiseCov(const MatrixXt& m)
  {
    measurement_noise_ = m;
    return *this;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  const int state_dim_;
  const int input_dim_;
  const int measurement_dim_;

  const int N_;
  const int M_;
  const int K_;
  const int S_;

public:
  VectorXt mean_;
  MatrixXt cov_;

  System system_;
  MatrixXt process_noise_;      //
  MatrixXt measurement_noise_;  //

  T lambda_;
  VectorXt weights_;

  MatrixXt sigma_points_;

  VectorXt ext_weights_;
  MatrixXt ext_sigma_points_;
  MatrixXt expected_measurements_;

private:
  /**
   * @brief compute sigma points
   * @param mean          mean
   * @param cov           covariance
   * @param sigma_points  calculated sigma points
   */
  void computeSigmaPoints(const VectorXt& mean, const MatrixXt& cov, MatrixXt& sigma_points)
  {
    const int n = mean.size();
    assert(cov.rows() == n && cov.cols() == n);

    Eigen::LLT<MatrixXt> llt;
    llt.compute((n + lambda) * cov);
    MatrixXt l = llt.matrixL();

    sigma_points.row(0) = mean;
    for (int i = 0; i < n; i++)
    {
      sigma_points.row(1 + i * 2) = mean + l.col(i);
      sigma_points.row(1 + i * 2 + 1) = mean - l.col(i);
    }
  }

  /**
   * @brief make covariance matrix positive finite
   * @param cov  covariance matrix
   */
  void ensurePositiveFinite(MatrixXt& cov)
  {
    return;
    const double eps = 1e-9;

    Eigen::EigenSolver<MatrixXt> solver(cov);
    MatrixXt d = solver.pseudoEigenvalueMatrix();
    MatrixXt v = solver.pseudoEigenvectors();
    for (int i = 0; i < d.rows(); i++)
    {
      if (d(i, i) < eps)
      {
        d(i, i) = eps;
      }
    }

    cov = v * d * v.inverse();
  }

public:
  MatrixXt kalman_gain_;

  std::mt19937 mt_;
  std::normal_distribution<T> normal_dist_;
};

}  // namespace alg
}  // namespace kkl

#endif
