#include "structure_tensors/st_via_convolution.h"

namespace structure_tensors { 

void compute_eigen_values(cv::Mat const& tensors, 
                          cv::Mat& eigens, 
                          cv::Mat& img_max_eigen_vectors,
                          cv::Mat& img_min_eigen_vectors)
{
  for (size_t x = 0; x < tensors.cols; x++)
  {
    for (size_t y = 0; y < tensors.rows; y++)
    {
      cv::Mat t = (cv::Mat_<float>(2, 2) << tensors.at<cv::Vec3f>(y, x)[0], tensors.at<cv::Vec3f>(y, x)[2],
                            tensors.at<cv::Vec3f>(y, x)[2], tensors.at<cv::Vec3f>(y, x)[1]);
      cv::Mat eigen_values;
      cv::Mat eigen_vectors;
      cv::eigen(t, eigen_values, eigen_vectors);

      eigens.at<cv::Vec2f>(y, x)[0] = eigen_values.at<float>(0, 0);
      eigens.at<cv::Vec2f>(y, x)[1] = eigen_values.at<float>(1, 0);    

      img_max_eigen_vectors.at<cv::Vec2f>(y, x)[0] = eigen_vectors.at<float>(0, 0);
      img_max_eigen_vectors.at<cv::Vec2f>(y, x)[1] = eigen_vectors.at<float>(0, 1);

      img_min_eigen_vectors.at<cv::Vec2f>(y, x)[0] = eigen_vectors.at<float>(1, 0);
      img_min_eigen_vectors.at<cv::Vec2f>(y, x)[1] = eigen_vectors.at<float>(1, 1);

      if (eigens.at<cv::Vec2f>(y, x)[0] < eigens.at<cv::Vec2f>(y, x)[1])
      {
        std::swap(eigens.at<cv::Vec2f>(y, x)[0], eigens.at<cv::Vec2f>(y, x)[1]);
        std::swap(img_max_eigen_vectors.at<cv::Vec2f>(y, x)[0], img_max_eigen_vectors.at<cv::Vec2f>(y, x)[1]);
        std::swap(img_min_eigen_vectors.at<cv::Vec2f>(y, x)[0], img_min_eigen_vectors.at<cv::Vec2f>(y, x)[1]);
      }
    }
  }
}


cv::Mat min_eigens(cv::Mat const& eigens)
{
  cv::Mat min_eigens(eigens.rows, eigens.cols, CV_32FC1, cv::Scalar(0));
  for (size_t x = 0; x < eigens.cols; x++)
  {
    for (size_t y = 0; y < eigens.rows; y++)
    {
      min_eigens.at<float>(y,x) = eigens.at<cv::Vec2f>(y, x)[1];
    }
  }
  cv::normalize(min_eigens, min_eigens, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::imwrite("min_eigens.jpg", min_eigens);
  return min_eigens;
}


cv::Mat relation_of_min_and_max_eigens(cv::Mat const& eigens)
{
  cv::Mat min_to_max(eigens.rows, eigens.cols, CV_32FC1, cv::Scalar(0));
  for (size_t x = 0; x < eigens.cols; x++)
  {
    for (size_t y = 0; y < eigens.rows; y++)
    {
      min_to_max.at<float>(y, x) = eigens.at<cv::Vec2f>(y, x)[1] / eigens.at<cv::Vec2f>(y, x)[0];
    }
  }
  cv::normalize(min_to_max, min_to_max, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::imwrite("min_to_max.jpg", min_to_max);
  return min_to_max;
}


void first_derivatives(cv::Mat const& img,
                       int const sobel_size,
                       cv::Mat& dx,
                       cv::Mat& dy)
{
  cv::Sobel(img, dx, CV_32F, 1, 0, sobel_size);// sobel_size = 1,3,5,7
  cv::Sobel(img, dy, CV_32F, 0, 1, sobel_size);
  /*cv::Mat u_kernel(cv::Size(2, 2), CV_32F, { 1, 0, 0, -1 });
  cv::filter2D(img, dx, -1, u_kernel);

  cv::Mat v_kernel(cv::Size(2, 2), CV_32F, { 0, 1, -1, 0 });
  cv::filter2D(img, dy, -1, v_kernel);*/
}


void squares_derivatives(cv::Mat const& dx,
                         cv::Mat const& dy,
                         cv::Mat& dx_dx,
                         cv::Mat& dy_dy,
                         cv::Mat& dx_dy)
{
  cv::pow(dx, 2, dx_dx);
  cv::pow(dy, 2, dy_dy);
  dx_dy = dx.mul(dy);
}


void mean_first_derivatives(cv::Mat const& dx,
                            cv::Mat const& dy,
                            cv::Size const mean_filter_size,
                            std::vector<cv::Mat>& mean_first_deriv) 
{
  cv::boxFilter(dx, mean_first_deriv[0], -1, mean_filter_size, cv::Point(-1, -1), true);
  cv::boxFilter(dy, mean_first_deriv[1], -1, mean_filter_size, cv::Point(-1, -1), true);
}


void mean_squares_derivatives(cv::Mat const& dx_dx,
                              cv::Mat const& dy_dy,
                              cv::Mat const& dx_dy,
                              cv::Size const mean_filter_size,
                              std::string const& window_shape,
                              std::vector<cv::Mat>& mean_sqr_deriv)
{
  if (window_shape.compare("BOX") == 0) 
  {
    cv::boxFilter(dx_dx, mean_sqr_deriv[0], -1, mean_filter_size, cv::Point(-1, -1), true);
    cv::boxFilter(dy_dy, mean_sqr_deriv[1], -1, mean_filter_size, cv::Point(-1, -1), true);
    cv::boxFilter(dx_dy, mean_sqr_deriv[2], -1, mean_filter_size, cv::Point(-1, -1), true);
  }
  else 
  {
    if (window_shape.compare("GAUSSIAN") == 0)
    {
      cv::GaussianBlur(dx_dx, mean_sqr_deriv[0], cv::Size(0, 0), mean_filter_size.width, mean_filter_size.height);
      cv::GaussianBlur(dy_dy, mean_sqr_deriv[1], cv::Size(0, 0), mean_filter_size.width, mean_filter_size.height);
      cv::GaussianBlur(dx_dy, mean_sqr_deriv[2], cv::Size(0, 0), mean_filter_size.width, mean_filter_size.height);
    }
    else
    {
      throw std::runtime_error("window_shape must be BOX or GAUSSIAN");
    }
  }
}


cv::Mat structure_tensors_via_convolutions(cv::Mat const& img,
                                           int const sobel_size,
                                           cv::Size const mean_filter_size,
                                           std::string const& window_shape)
{
  cv::Mat dx;
  cv::Mat dy;
  cv::Mat dx_dx;
  cv::Mat dy_dy;
  cv::Mat dx_dy;
  first_derivatives(img, sobel_size, dx, dy);
  squares_derivatives(dx, dy, dx_dx, dy_dy, dx_dy);
  //t_0_0 t_0_1
  //t_1_0 t_1_1                           //dx_sq_sum //dy_sq_sum //dx_dy_sum
  std::vector<cv::Mat> mean_sqr_deriv(3); //t_0_0     //t_1_1     //t_0_1
  
  mean_squares_derivatives(dx_dx, dy_dy, dx_dy, mean_filter_size, window_shape, mean_sqr_deriv);
  
  cv::Mat structure_tensors(img.rows, img.cols, CV_32FC3, cv::Scalar(0, 0));
  cv::merge(mean_sqr_deriv, structure_tensors);
  
  return structure_tensors;
}


cv::Mat centric_structure_tensors_via_convolutions(cv::Mat const& img,
                                                   int const sobel_size,
                                                   cv::Size const mean_filter_size,
                                                   std::string const& window_shape)
{
  cv::Mat dx;
  cv::Mat dy;
  cv::Mat dx_dx;
  cv::Mat dy_dy;
  cv::Mat dx_dy;
  first_derivatives(img, sobel_size, dx, dy);
  squares_derivatives(dx, dy, dx_dx, dy_dy, dx_dy);
  
  std::vector<cv::Mat> mean_sqr_deriv(3);
  
  mean_squares_derivatives(dx_dx, dy_dy, dx_dy, mean_filter_size, window_shape, mean_sqr_deriv);

  cv::Mat const& mean_dx_dx = mean_sqr_deriv[0];
  cv::Mat const& mean_dy_dy = mean_sqr_deriv[1];
  cv::Mat const& mean_dx_dy = mean_sqr_deriv[2];

  std::vector<cv::Mat> mean_first_deriv(2);
  mean_first_derivatives(dx, dy, mean_filter_size, mean_first_deriv);
  cv::Mat const& mean_dx = mean_first_deriv[0];
  cv::Mat const& mean_dy = mean_first_deriv[1];
  
  cv::Mat mean_dx_mean_dx;
  cv::Mat mean_dy_mean_dy;
  cv::Mat mean_dx_mean_dy;
  squares_derivatives(mean_dx, mean_dy, mean_dx_mean_dx, mean_dy_mean_dy, mean_dx_mean_dy);

  //t_0_0 t_0_1
  //t_1_0 t_1_1
  std::vector<cv::Mat> diff_mean_sqr_deriv(3);  //t_0_0     //t_1_1     //t_0_1

  mean_squares_derivatives(mean_dx_dx - mean_dx_mean_dx,
                           mean_dy_dy - mean_dy_mean_dy,
                           mean_dx_dy - mean_dx_mean_dy,
                           mean_filter_size,
                           window_shape,
                           diff_mean_sqr_deriv);

  cv::Mat centric_structure_tensors;
  cv::merge(diff_mean_sqr_deriv, centric_structure_tensors);

  return centric_structure_tensors;
}


cv::Mat structure_tensors(cv::Mat const& img,
                          int const sobel_size,
                          cv::Size const mean_filter_size,
                          std::string const& window_shape,
                          bool const is_centric,
                          int const drawing_step,
                          float const drawing_scale)
{
  cv::Mat structure_tensors;
  if (is_centric)
  {
    structure_tensors = centric_structure_tensors_via_convolutions(img,
      sobel_size,
      mean_filter_size,
      window_shape);
  }
  else 
  {
    structure_tensors = structure_tensors_via_convolutions(img,
      sobel_size,
      mean_filter_size,
      window_shape);
  }

  return structure_tensors; 
}} // ns structure_tensors