#ifndef POSE_SYSTEM_HPP
#define POSE_SYSTEM_HPP

#include <kkl/alg/unscented_kalman_filter.hpp>

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
  PoseSystem()
  {
    dt_ = 0.01;
  }

  // system equation (without input)
  VectorXt f(const VectorXt& state) const
  {
    VectorXt state_next(16);

    Vector3t pt = state.middleRows(0, 3);
    Vector3t vt = state.middleRows(3, 3);
    Quaterniont qt(state[6], state[7], state[8], state[9]);
    qt.normalize();

    Vector3t acc_bias = state.middleRows(10, 3);
    Vector3t gyro_bias = state.middleRows(13, 3);

    // position
    state_next.middleRows(0, 3) = pt + vt * dt_;  //

    // velocity
    state_next.middleRows(3, 3) = vt;

    // orientation
    state_next.middleRows(6, 4) << qt.w(), qt.x(), qt.y(), qt.z();
    state_next.middleRows(10, 3) = state.middleRows(10, 3);  // constant bias on acceleration
    state_next.middleRows(13, 3) = state.middleRows(13, 3);  // constant bias on angular velocity

    return state_next;
  }

  // system equation
  VectorXt fImu(const VectorXt& state, const Eigen::Vector3f& imu_acc, const Eigen::Vector3f& imu_gyro) const
  {
    VectorXt state_next(16);

    Vector3t pt = state.middleRows(0, 3);
    Vector3t vt = state.middleRows(3, 3);
    Quaterniont qt(state[6], state[7], state[8], state[9]);
    qt.normalize();

    Vector3t acc_bias = state.middleRows(10, 3);
    Vector3t gyro_bias = state.middleRows(13, 3);

    const Vector3t& raw_acc = imu_acc;
    const Vector3t& raw_gyro = imu_gyro;

    // position
    state_next.middleRows(0, 3) = pt + vt * dt_;  //

    // velocity
    Vector3t g(0.0f, 0.0f, 9.80665f);
    Vector3t acc = qt * (raw_acc - acc_bias);
    state_next.middleRows(3, 3) = vt + (acc - g) * dt_;
    // state_next.middleRows(3, 3) = vt; // + (acc - g) * dt_;		// acceleration didn't contribute to accuracy due to
    // large noise

    // orientation
    Vector3t gyro = raw_gyro - gyro_bias;
    Quaterniont dq(1, gyro[0] * dt_ / 2, gyro[1] * dt_ / 2, gyro[2] * dt_ / 2);
    dq.normalize();
    Quaterniont qt_next = (qt * dq).normalized();
    state_next.middleRows(6, 4) << qt_next.w(), qt_next.x(), qt_next.y(), qt_next.z();

    state_next.middleRows(10, 3) = state.middleRows(10, 3);  // constant bias on acceleration
    state_next.middleRows(13, 3) = state.middleRows(13, 3);  // constant bias on angular velocity

    return state_next;
  }

  VectorXt fOdom(const VectorXt& state, const Eigen::Vector3f& odom_twist_lin,
                 const Eigen::Vector3f& odom_twist_ang) const
  {
    VectorXt state_next(16);
    Vector3t pt = state.middleRows(0, 3);
    Vector3t vt = state.middleRows(3, 3);
    Quaterniont qt(state[6], state[7], state[8], state[9]);
    qt.normalize();

    Vector3t lin_vel_raw = odom_twist_lin;
    Vector3t ang_vel_raw = odom_twist_ang;

    // position
    state_next.middleRows(0, 3) = pt + vt * dt_;

    // velocity
    Vector3t vel = qt * lin_vel_raw;
    state_next.middleRows(3, 3) = vel;

    // orientation
    Quaterniont dq(1, ang_vel_raw[0] * dt_ / 2, ang_vel_raw[1] * dt_ / 2, ang_vel_raw[2] * dt_ / 2);
    dq.normalize();
    Quaterniont qt_next = (qt * dq).normalized();
    state_next.middleRows(6, 4) << qt_next.w(), qt_next.x(), qt_next.y(), qt_next.z();

    state_next.middleRows(10, 3) = state.middleRows(10, 3);  // constant bias on acceleration
    state_next.middleRows(13, 3) = state.middleRows(13, 3);  // constant bias on angular velocity

    return state_next;
  }

  // observation equation
  VectorXt h(const VectorXt& state) const
  {
    VectorXt observation(7);
    observation.middleRows(0, 3) = state.middleRows(0, 3);
    observation.middleRows(3, 4) = state.middleRows(6, 4).normalized();

    return observation;
  }

  double dt_;
};

}  // namespace hdl_localization

#endif  // POSE_SYSTEM_HPP
