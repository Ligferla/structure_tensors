#pragma once

#include <structure_tensors/st_weird.h>

namespace structure_tensors {

int calculate_diagonal_tensor_element(cv::Mat const& img,
                                      cv::Mat const& img_area,
                                      cv::Mat const& sq_img_area,
                                      cv::Point const& p,
                                      int const k)
{
  float const two_hat_I_norm = sq_img_area.at<float>(p.y, p.x) / k;

  float const one_hat_I = img_area.at<float>(p.y, p.x);
  float const one_hat_I_norm_sq = (one_hat_I / k) * (one_hat_I / k);

  float const I = img.at<float>(p.y, p.x);

  float I_and_one_hat_I_diff_norm_sq = ((k + 1) * I - one_hat_I) / k;
  I_and_one_hat_I_diff_norm_sq *= I_and_one_hat_I_diff_norm_sq;

  int const x_0_0 = two_hat_I_norm - one_hat_I_norm_sq + I_and_one_hat_I_diff_norm_sq;
  return x_0_0;
}


float calculate_another_tensor_element(float const t_0_0,
                                       float const t_1_1,
                                       cv::Mat const& x_img_area,
                                       cv::Mat const& x_sq_img_area,
                                       cv::Mat const& y_img_area,
                                       cv::Mat const& y_sq_img_area,
                                       cv::Point const& p,
                                       int const k)
{ // t_0_1 // t_1_0
  float const x_one_hat_I = x_img_area.at<float>(p.y, p.x);
  float const x_two_hat_I = x_sq_img_area.at<float>(p.y, p.x);
  float const y_one_hat_I = y_img_area.at<float>(p.y, p.x);
  float const y_two_hat_I = y_sq_img_area.at<float>(p.y, p.x);

  float const sq_diff = x_two_hat_I + y_two_hat_I
    - (1 / k) * (x_one_hat_I*x_one_hat_I
      + y_one_hat_I*y_one_hat_I
      + (y_one_hat_I - x_one_hat_I)*(y_one_hat_I - x_one_hat_I));

  return (1 / 2) * (t_0_0 + t_1_1 + sq_diff);
}


cv::Mat weird_structure_tensors(cv::Mat const& img, int const mean_filter_size)
{
  cv::Mat sq_img;
  cv::pow(img, 2, sq_img);

  int const k = static_cast<int>(mean_filter_size * mean_filter_size / 2);
  cv::Mat x_img_area;
  cv::boxFilter(img, x_img_area, -1, cv::Size(mean_filter_size, mean_filter_size / 2), cv::Point(-1, -1), false);

  cv::Mat x_sq_img_area;
  cv::boxFilter(sq_img, x_sq_img_area, -1, cv::Size(mean_filter_size, mean_filter_size / 2), cv::Point(-1, -1), false);

  cv::Mat y_img_area;
  cv::boxFilter(img, y_img_area, -1, cv::Size(mean_filter_size / 2, mean_filter_size), cv::Point(-1, -1), false);

  cv::Mat y_sq_img_area;
  cv::boxFilter(sq_img, y_sq_img_area, -1, cv::Size(mean_filter_size / 2, mean_filter_size), cv::Point(-1, -1), false);

  cv::Mat image_structure_tensor(img.rows, img.cols, CV_32FC3, cv::Scalar(0, 0, 0));
  for (size_t y = 0; y < img.rows; y++)
  {
    for (size_t x = 0; x < img.cols; x++)
    {
      //t_0_0 t_0_1
      //t_1_0 t_1_1

      //t_0_0 
      image_structure_tensor.at<cv::Vec3f>(y, x)[0] = calculate_diagonal_tensor_element(img, x_img_area, x_sq_img_area, cv::Point(x, y), k);
      float const t_0_0 = image_structure_tensor.at<cv::Vec3f>(y, x)[0];
      //t_1_1 
      image_structure_tensor.at<cv::Vec3f>(y, x)[1] = calculate_diagonal_tensor_element(img, y_img_area, y_sq_img_area, cv::Point(x, y), k);
      float const t_1_1 = image_structure_tensor.at<cv::Vec3f>(y, x)[1];
      //t_1_0 
      image_structure_tensor.at<cv::Vec3f>(y, x)[2] = calculate_another_tensor_element(t_0_0, t_1_1, x_img_area, x_sq_img_area, y_img_area, y_sq_img_area, cv::Point(x, y), k);
    }
  }

  return image_structure_tensor;
}} // ns structure_tensors