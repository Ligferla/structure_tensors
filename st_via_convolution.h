#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

namespace structure_tensors {

void compute_eigen_values(cv::Mat const& tensors, 
                          cv::Mat& eigens, 
                          cv::Mat& img_max_eigen_vectors,
                          cv::Mat& img_min_eigen_vectors);

cv::Mat min_eigens(cv::Mat const& eigens);

cv::Mat relation_of_min_and_max_eigens(cv::Mat const& eigens);

void first_derivatives(cv::Mat const& img,
                       int const sobel_size,
                       cv::Mat& dx,
                       cv::Mat& dy);

void squares_derivatives(cv::Mat const& dx,
                         cv::Mat const& dy,
                         cv::Mat& dx_dx,
                         cv::Mat& dy_dy,
                         cv::Mat& dx_dy);

void mean_first_derivatives(cv::Mat const& dx,
                            cv::Mat const& dy,
                            cv::Size const mean_filter_size,
                            std::vector<cv::Mat>& mean_first_deriv);

void mean_squares_derivatives(cv::Mat const& dx_dx,
                              cv::Mat const& dy_dy,
                              cv::Mat const& dx_dy,
                              cv::Size const mean_filter_size,
                              std::string const& window_shape,
                              std::vector<cv::Mat>& mean_sqr_deriv);

cv::Mat structure_tensors_via_convolutions(cv::Mat const& img,
                                           int const sobel_size,
                                           int const mean_filter_size,
                                           std::string const& window_shape);

cv::Mat centric_structure_tensors_via_convolutions(cv::Mat const& img,
                                                   int const sobel_size,
                                                   int const mean_filter_size,
                                                   std::string const& window_shape);

cv::Mat structure_tensors(cv::Mat const& img,
                          int const sobel_size,
                          cv::Size const mean_filter_size,
                          std::string const& window_shape,
                          bool const is_centric,
                          int const drawing_step,
                          float const drawing_scale);
} // ns structure_tensors