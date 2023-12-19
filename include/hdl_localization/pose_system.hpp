// pose_system.hpp
#ifndef POSE_SYSTEM_HPP
#define POSE_SYSTEM_HPP

#include <ukf/unscented_kalman_filter.hpp>

namespace hdl_localization
{
/**
 * @brief Definition of system to be estimated by ukf
 * @note state = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, acc_bias_x, acc_bias_y, acc_bias_z, gyro_bias_x, gyro_bias_y,
 * gyro_bias_z]
 */
class PoseSystem
{
public:
  typedef float T;
  typedef Eigen::Matrix<T, 3, 1> Vector3t;
  typedef Eigen::Matrix<T, 4, 4> Matrix4t;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  typedef Eigen::Quaternion<T> Quaterniont;

public:
  PoseSystem();
  // system equation (without input)
  VectorXt f(const VectorXt& state) const;
  // system equation
  VectorXt fImu(const VectorXt& state, const Eigen::Vector3f& imu_acc, const Eigen::Vector3f& imu_gyro) const;
  VectorXt fOdom(const VectorXt& state, const Eigen::Vector3f& odom_twist_lin,
                 const Eigen::Vector3f& odom_twist_ang) const;
  // observation equation
  VectorXt h(const VectorXt& state) const;

  double dt_;
};

}  // namespace hdl_localization

#endif  // POSE_SYSTEM_HPP
