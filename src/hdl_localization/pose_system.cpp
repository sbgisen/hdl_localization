#include <hdl_localization/pose_system.hpp>

namespace hdl_localization
{
PoseSystem::PoseSystem()
{
  dt_ = 0.01;
}

// system equation (without input)
PoseSystem::VectorXt PoseSystem::f(const VectorXt& state) const
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
PoseSystem::VectorXt PoseSystem::fImu(const VectorXt& state, const Eigen::Vector3f& imu_acc,
                                      const Eigen::Vector3f& imu_gyro) const
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

PoseSystem::VectorXt PoseSystem::fOdom(const VectorXt& state, const Eigen::Vector3f& odom_twist_lin,
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
PoseSystem::VectorXt PoseSystem::h(const VectorXt& state) const
{
  VectorXt observation(7);
  observation.middleRows(0, 3) = state.middleRows(0, 3);
  observation.middleRows(3, 4) = state.middleRows(6, 4).normalized();

  return observation;
}

}  // namespace hdl_localization
