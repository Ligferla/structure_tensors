#include "structure_tensors/st_visualization.h"

namespace structure_tensors {

cv::Mat draw_eigen_vectors(cv::Mat const& img,
                           cv::Mat const& eigens,
                           cv::Mat const& img_max_eigen_values,
                           cv::Mat const& img_min_eigen_values,
                           int const drawing_step,
                           float const drawing_scale/*,
                           int const mean_filter_size*/)
{
  cv::Mat img_with_eigen_vectors = img.clone();
  img_with_eigen_vectors.convertTo(img_with_eigen_vectors, CV_8U);
  cv::cvtColor(img_with_eigen_vectors, img_with_eigen_vectors, CV_GRAY2RGB);

  for (size_t y = 0; y < img_with_eigen_vectors.rows; y += drawing_step)
  {
    for (size_t x = 0; x < img_with_eigen_vectors.cols; x += drawing_step)
    {
      cv::Point pt1(x, y);
      float const semiminor_axis = abs(eigens.at<cv::Vec2f>(y, x)[0]);
      float const semimajor_axis = abs(eigens.at<cv::Vec2f>(y, x)[1]);
      cv::Point2f max_eigen_vector(img_max_eigen_values.at<cv::Vec2f>(y, x)[0],  
                                   img_max_eigen_values.at<cv::Vec2f>(y, x)[1]);
      cv::Point2f min_eigen_vector(img_min_eigen_values.at<cv::Vec2f>(y, x)[0], 
                                   img_min_eigen_values.at<cv::Vec2f>(y, x)[1]);

      double const angle = std::atan(max_eigen_vector.y / max_eigen_vector.x) * 180 / M_PI;
      cv::ellipse(img_with_eigen_vectors, cv::Point(x, y), cv::Size(drawing_scale * semiminor_axis, drawing_scale * semimajor_axis), angle, 0, 360, cv::Scalar(0, 255, 0), 1);
    }
  }

  /*cv::Point pt1(img.cols / 2 - mean_filter_size / 2, img.rows / 2 - mean_filter_size / 2);
  cv::Point pt2(img.cols / 2 + mean_filter_size / 2, img.rows / 2 + mean_filter_size / 2);
  cv::rectangle(img_with_eigen_vectors, pt1, pt2, cv::Scalar(255, 0, 0), 2);*/

  return img_with_eigen_vectors;
}}// ns structure_tensors