#ifndef POSE_SYSTEM_HPP
#define POSE_SYSTEM_HPP

#include <kkl/alg/unscented_kalman_filter.hpp>

namespace hdl_localization {

/**
 * @brief Definition of system to be estimated by ukf
 * @note state = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, acc_bias_x, acc_bias_y, acc_bias_z, gyro_bias_x, gyro_bias_y, gyro_bias_z]
 */
class PoseSystem {
public:
  typedef float T;
  typedef Eigen::Matrix<T, 3, 1> Vector3t;
  typedef Eigen::Matrix<T, 4, 4> Matrix4t;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  typedef Eigen::Quaternion<T> Quaterniont;

public:
  PoseSystem() { dt = 0.01; }

  // system equation (without input)
  VectorXt f(const VectorXt& state) const {
    VectorXt next_state(16);

    Vector3t pt = state.middleRows(0, 3);
    Vector3t vt = state.middleRows(3, 3);
    Quaterniont qt(state[6], state[7], state[8], state[9]);
    qt.normalize();

    Vector3t acc_bias = state.middleRows(10, 3);
    Vector3t gyro_bias = state.middleRows(13, 3);

    // position
    next_state.middleRows(0, 3) = pt + vt * dt;  //

    // velocity
    next_state.middleRows(3, 3) = vt;

    // orientation
    Quaterniont qt_ = qt;

    next_state.middleRows(6, 4) << qt_.w(), qt_.x(), qt_.y(), qt_.z();
    next_state.middleRows(10, 3) = state.middleRows(10, 3);  // constant bias on acceleration
    next_state.middleRows(13, 3) = state.middleRows(13, 3);  // constant bias on angular velocity

    return next_state;
  }

  // system equation
  VectorXt f_imu(const VectorXt& state, const Eigen::Vector3f& imu_acc, const Eigen::Vector3f& imu_gyro) const {
    VectorXt next_state(16);

    Vector3t pt = state.middleRows(0, 3);
    Vector3t vt = state.middleRows(3, 3);
    Quaterniont qt(state[6], state[7], state[8], state[9]);
    qt.normalize();

    Vector3t acc_bias = state.middleRows(10, 3);
    Vector3t gyro_bias = state.middleRows(13, 3);

    // position
    Vector3t next_pt = pt + vt * dt;
    next_state.middleRows(0, 3) = next_pt;

    // velocity (vel_z = 0);
    Vector3t g(0.0f, 0.0f, 9.80665f);
    Vector3t acc = qt * (imu_acc - acc_bias - g);
    Vector3t next_vt = vt + acc * dt;
    next_vt.z() = 0.0f;
    next_state.middleRows(3, 3) = next_vt;  // acceleration didn't contribute to accuracy due to large noise

    // orientation
    Vector3t gyro = imu_gyro - gyro_bias;
    Quaterniont dq(1, gyro.x() * dt, gyro.y() * dt, gyro.z() * dt);
    dq.normalize();
    Quaterniont next_qt = (qt * dq).normalized();
    next_state.middleRows(6, 4) << next_qt.w(), next_qt.x(), next_qt.y(), next_qt.z();
    next_state.middleRows(10, 3) = state.middleRows(10, 3);  // constant bias on acceleration
    next_state.middleRows(13, 3) = state.middleRows(13, 3);  // constant bias on angular velocity

    return next_state;
  }

  VectorXt f_odom(const VectorXt& state, const Eigen::Vector3f& odom_twist_lin, const Eigen::Vector3f& odom_twist_ang) const {
    VectorXt next_state(16);
    Vector3t pt = state.middleRows(0, 3);
    Vector3t vt = state.middleRows(3, 3);
    Quaterniont qt(state[6], state[7], state[8], state[9]);
    qt.normalize();

    Vector3t raw_lin_vel = odom_twist_lin;
    Vector3t raw_ang_vel = odom_twist_ang;

    // position
    next_state.middleRows(0, 3) = pt + vt * dt;

    // velocity (vel_z = 0);
    Vector3t vel = qt * raw_lin_vel;
    vel.z() = 0.0f;
    next_state.middleRows(3, 3) = vel;

    // orientation
    Quaterniont dq(1, raw_ang_vel[0] * dt / 2, raw_ang_vel[1] * dt / 2, raw_ang_vel[2] * dt / 2);
    dq.normalize();
    Quaterniont qt_ = (qt * dq).normalized();
    next_state.middleRows(6, 4) << qt_.w(), qt_.x(), qt_.y(), qt_.z();

    next_state.middleRows(10, 3) = state.middleRows(10, 3);  // constant bias on acceleration
    next_state.middleRows(13, 3) = state.middleRows(13, 3);  // constant bias on angular velocity

    return next_state;
  }

  // observation equation
  VectorXt h(const VectorXt& state) const {
    VectorXt observation(7);
    observation.middleRows(0, 3) = state.middleRows(0, 3);
    observation.middleRows(3, 4) = state.middleRows(6, 4).normalized();

    return observation;
  }

  double dt;
};

}  // namespace hdl_localization

#endif  // POSE_SYSTEM_HPP
