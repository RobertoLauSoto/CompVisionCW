// header inclusion
#include "convolution.h"

#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

void convolution(const cv::Mat_<uchar> &input, cv::Mat_<float> &kernel, cv::Mat_<float> &output){
	output.create(input.size());
	int kernelRadiusX = (kernel.size[0] - 1) / 2;
	int kernelRadiusY = (kernel.size[1] - 1) / 2;

	cv::Mat_<uchar> padded_input;
	cv::copyMakeBorder(input, padded_input, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, cv::BORDER_REPLICATE);

	for (int i = 0; i < input.rows; i++){
		for (int j = 0; j < input.cols; j++){
			float sum = 0.0;
			for (int m = -kernelRadiusX; m <= kernelRadiusX; m++){
				for (int n = -kernelRadiusY; n <= kernelRadiusY; n++){
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;
					int imageval = padded_input(imagex, imagey);
					float kernalval = kernel(kernelx, kernely);
					sum += imageval * kernalval;
				}
			}
			output(i, j) = sum;
		}
	}
}