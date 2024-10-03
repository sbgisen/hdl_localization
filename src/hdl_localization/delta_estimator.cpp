#include "hdl_localization/delta_estimator.hpp"

namespace hdl_localization
{
DeltaEstimator::DeltaEstimator(pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr reg)
  : delta_(Eigen::Isometry3f::Identity()), reg_(reg)
{
}

DeltaEstimator::~DeltaEstimator()
{
}

void DeltaEstimator::reset()
{
  std::lock_guard<std::mutex> lock(mutex_);
  delta_.setIdentity();
  last_frame_.reset();
}

void DeltaEstimator::addFrame(pcl::PointCloud<PointT>::ConstPtr frame)
{
  std::unique_lock<std::mutex> lock(mutex_);
  if (last_frame_ == nullptr)
  {
    last_frame_ = frame;
    return;
  }

  reg_->setInputTarget(last_frame_);
  reg_->setInputSource(frame);
  lock.unlock();

  pcl::PointCloud<PointT> aligned;
  reg_->align(aligned);

  lock.lock();
  last_frame_ = frame;
  delta_ = delta_ * Eigen::Isometry3f(reg_->getFinalTransformation());
}

Eigen::Isometry3f DeltaEstimator::estimatedDelta() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return delta_;
}

}  // namespace hdl_localization
