#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace structure_tensors {

  cv::Mat draw_eigen_vectors(cv::Mat const& img,
                             cv::Mat const& eigens,
                             cv::Mat const& img_max_eigen_values,
                             cv::Mat const& img_min_eigen_values,
                             int const drawing_step,
                             float const drawing_scale/*,
                             int const mean_filter_size*/);
}// ns structure_tensors