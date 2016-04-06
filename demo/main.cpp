#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <json-cpp/json.h>
#include <mxprops/mxprops.h>
#include <mxprops/io.h>
#include "structure_tensors/st_via_convolution.h" 
#include "structure_tensors/st_weird.h"
#include "structure_tensors/st_visualization.h"
#include "structure_tensors/mean_and_standart_deviation.h"

namespace structure_tensors {

/*static cv::Mat generate_img(std::string const& filename)
{
  int const rows = 600;
  int const cols = 1000;
  cv::Mat img(rows, cols, CV_8UC1, cv::Scalar(255));
  int const step = static_cast<int>(cols / 50);

  for (size_t x = 3 * step; x < static_cast<int>(img.cols / 2) - 2 * step; x += step)
  {
    cv::Point pt1(x, 0);
    cv::Point pt2(x, img.rows);
    cv:line(img, pt1, pt2, cv::Scalar(0), 4, CV_AA);
  }

  int radius = static_cast<int>(step / 4);
  for (size_t y = 3 * step; y < img.rows - 2 * step; y += step)
  {
    for (size_t x = 3 * step + static_cast<int>(img.cols / 2); x < img.cols - 2 * step; x += step)
    {
      cv::circle(img, cv::Point(x,y), radius, cv::Scalar(0), -1);
    }
  }
  cv::imwrite(filename, img);
  return img;
}*/


static cv::Mat read_image(std::string const& path)
{
  //generate_img(path);
  cv::Mat result = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
  if (result.empty())
    throw std::runtime_error("Unable to read image: " + path);
  return result;
}


static Json::Value read_json(std::string const& json_path)
{
  Json::Value root;
  Json::Reader reader;
  std::ifstream ifs(json_path);
  if (!ifs.is_open())
    throw std::runtime_error("Unable to open json file " + json_path);

  if (reader.parse(ifs, root))
    return root;
  else
    throw std::runtime_error("Unable to read json file: " + reader.getFormatedErrorMessages());
}

static void main(mxprops::PTree::ConstRef const& props)
{
  cv::Mat src = read_image(props.get<std::string>("input_image_path"));

  src.convertTo(src, CV_32F);
  float const img_blur_radius = 1.5;
  cv::Mat blurred_img;
  cv::GaussianBlur(src, blurred_img, cv::Size(0, 0), img_blur_radius, img_blur_radius);

  // int mean_filter_size;
  int sobel_size;
  std::string window_shape;
  bool is_centric;
  int drawing_step;
  float drawing_scale;
  std::string output_image_path;

  // mean_filter_size = props.get<int>("alg_params.mean_filter_size", 40);
  sobel_size = props.get<int>("alg_params.sobel_size", 3);
  window_shape = props.get<std::string>("alg_params.window_shape", "BOX");
  is_centric = props.get<bool>("alg_params.is_centric", false);
  drawing_step = props.get<int>("alg_params.drawing_step", 20);
  drawing_scale = props.get<float>("alg_params.drawing_scale", 0.003);
// 
  // output_image_path = props.get<std::string>("output_image_path");

  //computation
  cv::Size mean_filter_size(blurred_img.cols, blurred_img.rows);
  cv::Mat st_tensors = structure_tensors(blurred_img,
    sobel_size,
    mean_filter_size,
    window_shape,
    is_centric,
    drawing_step,
    drawing_scale);

  //visualization
  cv::Mat eigens(st_tensors.rows, st_tensors.cols, CV_32FC2, cv::Scalar(0, 0));
  cv::Mat img_max_eigen_vectors(st_tensors.rows, st_tensors.cols, CV_32FC2, cv::Scalar(0, 0));
  cv::Mat img_min_eigen_vectors(st_tensors.rows, st_tensors.cols, CV_32FC2, cv::Scalar(0, 0));
  compute_eigen_values(st_tensors, eigens, img_max_eigen_vectors, img_min_eigen_vectors);

  cv::Mat img_with_eigen_vectors = draw_eigen_vectors(src,
    eigens, 
    img_max_eigen_vectors,
    img_min_eigen_vectors,
    drawing_step,
    drawing_scale/*,
    mean_filter_size*/);

  // cv::imwrite(output_image_path + "structure_tensors.jpg", img_with_eigen_vectors);
  
  //compute another characteristics
  //mean, standard deviation, min_intensity, max_intensity

  double min_intensity = 0;
  double max_intensity = 0;
  cv::minMaxLoc(blurred_img, &min_intensity, &max_intensity);

  /*cv::Mat mean;
  cv::Mat stddev;
  src.convertTo(src, CV_64F);
  cv::meanStdDev(blurred_img, mean, stddev);*/
  double mean = img_mean(blurred_img);
  double stand_dev = img_standart_deviation(blurred_img, mean);
  
  std::cout << st_tensors.at<cv::Vec3f>(st_tensors.rows / 2, st_tensors.cols / 2)[0] << " "
            << st_tensors.at<cv::Vec3f>(st_tensors.rows / 2, st_tensors.cols / 2)[1] << " "
            << st_tensors.at<cv::Vec3f>(st_tensors.rows / 2, st_tensors.cols / 2)[2] << " "
            << eigens.at<cv::Vec2f>(st_tensors.rows / 2, st_tensors.cols / 2)[0] << " "
            << eigens.at<cv::Vec2f>(st_tensors.rows / 2, st_tensors.cols / 2)[1] << " "
            << min_intensity << " "
            << max_intensity << " "
            << /*mean.at<float>(0,0)*/ mean << " "
            << /*stddev.at<float>(0,0)*/ stand_dev << std::endl;

  // // weird version
  // int wierd_mean_filter_size = 40;
  // cv::Mat weird_st_tensors = weird_structure_tensors(src, wierd_mean_filter_size);

  // //visualization of weird version
  // cv::Mat weird_eigens(weird_st_tensors.rows, weird_st_tensors.cols, CV_32FC2, cv::Scalar(0, 0));
  // cv::Mat img_max_weird_eigen_vectors(weird_st_tensors.rows, weird_st_tensors.cols, CV_32FC2, cv::Scalar(0, 0));
  // cv::Mat img_min_weird_eigen_vectors(weird_st_tensors.rows, weird_st_tensors.cols, CV_32FC2, cv::Scalar(0, 0));
  // compute_eigen_values(weird_st_tensors, weird_eigens, img_max_weird_eigen_vectors, img_min_weird_eigen_vectors);

  // cv::Mat img_with_weird_eigen_vectors = draw_eigen_vectors(src,
  //   weird_eigens, 
  //   img_max_weird_eigen_vectors,
  //   img_min_weird_eigen_vectors,
  //   drawing_step,
  //   drawing_scale/*,
  //   mean_filter_size*/);

  // cv::imwrite(output_image_path + "weird_structure_tensors.jpg", img_with_weird_eigen_vectors);

}} // ns structure_tensors


int main(int argc, char const *argv[])
{
  try
  {
    mxprops::PTree ptree;
    mxprops::PTree::Ref root = ptree.root("");
    mxprops::init_settings_from_command_line(root, argc, argv);
    structure_tensors::main(root);
    return 0;
  }
  catch (std::exception const& e)
  {
    std::cerr << "Unhandled exception: " << e.what() << "\n";
    return 1;
  }
  catch (...)
  {
    std::cerr << "Unhandled UNTYPED exception\n";
    return 2;
  }
}