#pragma once

#include <opencv2/core/core.hpp>

namespace structure_tensors {

	double img_mean(cv::Mat const& src);
	double img_standart_deviation(cv::Mat const& src, double const mean);

}// ns structure_tensors