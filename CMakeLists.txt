cmake_minimum_required(VERSION 2.8)

project( structure_tensors )

find_package( OpenCV REQUIRED )

add_library( structure_tensors
	     st_via_convolution.h
	     st_weird.h
	     st_visualization.h
	     mean_and_standart_deviation.h
	     src/structure_tensors.cpp
	     src/st_weird.cpp
	     src/st_visualization.cpp
	     src/mean_and_standart_deviation.cpp
	   )

target_link_libraries( structure_tensors ${OpenCV_LIBS})

add_executable( structure_tensors_test demo/main.cpp )
target_link_libraries( structure_tensors_test structure_tensors mxprops jsoncpp ${OpenCV_LIBS} ${Boost_LIBRARIES})
