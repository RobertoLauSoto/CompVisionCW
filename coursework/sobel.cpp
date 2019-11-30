#include "sobel.h"
#include <math.h>

void calc_mag(const cv::Mat_<float> &dx, const cv::Mat_<float> &dy, cv::Mat_<float> &output){	
	output.create(dx.size());
	for (int i = 0; i < dx.rows; i++){
		for (int j = 0; j < dx.cols; j++){
			float x = dx(i, j);
			float y = dy(i, j);
			output(i, j) = sqrt(x*x + y*y);
		}
	}
}

void calc_dir(const cv::Mat_<float> &dx, const cv::Mat_<float> &dy, cv::Mat_<float> &output){
	output.create(dx.size());
	for (int i = 0; i < dx.rows; i++){
		for (int j = 0; j < dx.cols; j++){
			float x = dx(i, j);
			float y = dy(i, j);
			output(i, j) = atan2(y, x);
		}
	}
}

void sobel(const cv::Mat_<uchar> &input, cv::Mat_<float> &dx_out, cv::Mat_<float> &dy_out, cv::Mat_<float> &mag_out, cv::Mat_<float> &dir_out){
	//Make kernels for dx & dy.
	cv::Mat_<float> dx_kernel(3, 3);
	dx_kernel << -1, 0, 1,
				-2, 0, 2,
				-1, 0, 1;

	cv::Mat_<float> dy_kernel(3, 3);
	dy_kernel << -1, -2, -1,
				 0,  0,  0,
				 1,  2,  1;

	//Calculate dx and dy.
	convolution(input, dx_kernel, dx_out);
	convolution(input, dy_kernel, dy_out);

	//Calculate magnitude
	calc_mag(dx_out, dy_out, mag_out);

	//Calculate directon
	calc_dir(dx_out, dy_out, dir_out);	
}

void blur(cv::Mat_<uchar> &input, cv::Mat_<float> &output){
	cv::Mat_<float> blur_kernel(3, 3);
	blur_kernel << 1.0/9.0, 1.0/9.0, 1.0/9.0,
				1.0/9.0, 1.0/9.0, 1.0/9.0,
				1.0/9.0, 1.0/9.0, 1.0/9.0;
	convolution(input, blur_kernel, output);
}