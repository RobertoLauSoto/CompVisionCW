#include "sobel.h"

#include <math.h>


void calculateMagnitude(const cv::Mat_<float> &dx, const cv::Mat_<float> &dy, cv::Mat_<float> &output){	
	assert(dx.size() == dy.size());
	output.create(dx.size());

	for (int i = 0; i < dx.rows; i++){
		for (int j = 0; j < dx.cols; j++){
			float x = dx(i, j);
			float y = dy(i, j);

			output(i, j) = sqrt(x*x + y*y);
		}
	}
}

void calculateDirection(const cv::Mat_<float> &dx, const cv::Mat_<float> &dy, cv::Mat_<float> &output){
	assert(dx.size() == dy.size());
	output.create(dx.size());

	for (int i = 0; i < dx.rows; i++){
		for (int j = 0; j < dx.cols; j++){
			float x = dx(i, j);
			float y = dy(i, j);

			output(i, j) = atan2(y, x);
		}
	}
}

void sobel(const cv::Mat_<uchar> &input, cv::Mat_<float> &dxOut, cv::Mat_<float> &dyOut, cv::Mat_<float> &magOut, cv::Mat_<float> &dirOut){
	//Make kernels for dx & dy.
	cv::Mat_<float> dxKernel(3, 3);
	dxKernel << -1, 0, 1,
				-2, 0, 2,
				-1, 0, 1;

	cv::Mat_<float> dyKernel(3, 3);
	dyKernel << -1, -2, -1,
				 0,  0,  0,
				 1,  2,  1;

	//Calculate dx & dy of image.
	convolution(input, dxKernel, dxOut);
	convolution(input, dyKernel, dyOut);

	//Calculate magnitude
	calculateMagnitude(dxOut, dyOut, magOut);

	//Calculate direction
	calculateDirection(dxOut, dyOut, dirOut);	
}

void blur(cv::Mat_<uchar> &input, cv::Mat_<float> &output){
	cv::Mat_<float> blurKernel(3, 3);
	blurKernel << 1.0/9.0, 1.0/9.0, 1.0/9.0,
				1.0/9.0, 1.0/9.0, 1.0/9.0,
				1.0/9.0, 1.0/9.0, 1.0/9.0;

	convolution(input, blurKernel, output);

}