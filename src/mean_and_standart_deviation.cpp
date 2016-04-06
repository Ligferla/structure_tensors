#include "structure_tensors/mean_and_standart_deviation.h"
#include "math.h"

namespace structure_tensors {

double img_mean(cv::Mat const& src)
{
  double mean = 0;
  for (size_t y = 0; y < src.rows; y++)
  {
    for (size_t x = 0; x < src.cols; x++)
    {
      mean += src.at<float>(y,x);
    }
  }
  mean /= src.cols * src.rows;
  return mean;
}


double img_standart_deviation(cv::Mat const& src, double const mean)
{
  double standart_deviation = 0;
  for (size_t y = 0; y < src.rows; y++)
  {
    for (size_t x = 0; x < src.cols; x++)
    {
      standart_deviation += (src.at<float>(y,x) - mean) * (src.at<float>(y,x) - mean);
    }
  }
  standart_deviation /= src.cols * src.rows;
  return std::sqrt(standart_deviation);
}
}// ns structure_tensors