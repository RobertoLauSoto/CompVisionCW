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

	cv::Mat_<uchar> paddedInput;
	cv::copyMakeBorder(input, paddedInput, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, cv::BORDER_REPLICATE);

	// now we can do the convoltion
	for (int i = 0; i < input.rows; i++){
		for (int j = 0; j < input.cols; j++){
			float sum = 0.0;
			for (int m = -kernelRadiusX; m <= kernelRadiusX; m++){
				for (int n = -kernelRadiusY; n <= kernelRadiusY; n++){
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = paddedInput(imagex, imagey);
					float kernalval = kernel(kernelx, kernely);

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
			output(i, j) = sum;
		}
	}
}