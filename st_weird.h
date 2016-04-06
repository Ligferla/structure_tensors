#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace structure_tensors {

int calculate_diagonal_tensor_element(cv::Mat const& img,
                                      cv::Mat const& img_area,
                                      cv::Mat const& sq_img_area,
                                      cv::Point const& p,
                                      int const k);

float calculate_another_tensor_element(float const t_0_0,
                                       float const t_1_1,
                                       cv::Mat const& x_img_area,
                                       cv::Mat const& x_sq_img_area,
                                       cv::Mat const& y_img_area,
                                       cv::Mat const& y_sq_img_area,
                                       cv::Point const& p,
                                       int const k);

cv::Mat weird_structure_tensors(cv::Mat const& img, int const mean_filter_size);
} // ns structure_tensors