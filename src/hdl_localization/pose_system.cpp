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

  // position
  state_next.middleRows(0, 3) = pt + vt * dt_;

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

  // position
  Vector3t pt_next = pt + vt * dt_;
  state_next.middleRows(0, 3) = pt_next;

  // velocity
  Vector3t g(0.0f, 0.0f, 9.80665f);
  Vector3t acc = qt * (imu_acc - acc_bias) - g;
  Vector3t vt_next = vt + acc * dt_;
  state_next.middleRows(3, 3) = vt_next;

  // orientation
  Vector3t gyro = imu_gyro - gyro_bias;
  Quaterniont dq;
  dq = Eigen::AngleAxisf(gyro[0] * dt_, Vector3t::UnitX()) * Eigen::AngleAxisf(gyro[1] * dt_, Vector3t::UnitY()) *
       Eigen::AngleAxisf(gyro[2] * dt_, Vector3t::UnitZ());
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

  // position
  Vector3t pt_next = pt + vt * dt_;
  state_next.middleRows(0, 3) = pt_next;

  // velocity
  Vector3t vt_next = qt * odom_twist_lin;
  state_next.middleRows(3, 3) = vt_next;

  // orientation
  Quaterniont dq;
  dq = Eigen::AngleAxisf(odom_twist_ang[0] * dt_, Vector3t::UnitX()) *
       Eigen::AngleAxisf(odom_twist_ang[1] * dt_, Vector3t::UnitY()) *
       Eigen::AngleAxisf(odom_twist_ang[2] * dt_, Vector3t::UnitZ());
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
