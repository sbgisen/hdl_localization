// hdl_localization_delta_estimator.hpp
#ifndef HDL_LOCALIZATION_DELTA_ESTIMATOR_HPP
#define HDL_LOCALIZATION_DELTA_ESTIMATOR_HPP

#include <mutex>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

namespace hdl_localization
{
class DeltaEstimator
{
public:
  using PointT = pcl::PointXYZI;

  DeltaEstimator(pcl::Registration<PointT, PointT>::Ptr reg);
  ~DeltaEstimator();

  void reset();
  void addFrame(pcl::PointCloud<PointT>::ConstPtr frame);
  Eigen::Isometry3f estimatedDelta() const;

private:
  mutable std::mutex mutex_;
  Eigen::Isometry3f delta_;
  pcl::Registration<PointT, PointT>::Ptr reg_;
  pcl::PointCloud<PointT>::ConstPtr last_frame_;
};

}  // namespace hdl_localization

#endif
