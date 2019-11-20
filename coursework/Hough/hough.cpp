#include <stdio.h>

#include <math.h>
#include <sobel.h>
#include <array.h>

#include <opencv2/opencv.hpp>        //you may need to
#include <opencv2/highgui.hpp>   //adjust import locations
#include <opencv2/imgproc.hpp> 
#include <opencv2/objdetect.hpp> 

using namespace cv;



void hough(const cv::Mat_<uchar> &input, cv::Mat_<float> &output, const uchar threshold, const int maxRadius, const float thetaThreshold)
{
	output.create(input.size());

	cv::Mat_<float> dx;
	cv::Mat_<float> dy;
	cv::Mat_<float> mag;
	cv::Mat_<float> dir;
	sobel(input, dx, dy, mag, dir);

	convertAndShow(dir, "directional");

	int ***H = malloc3dArray(input.rows, input.cols, maxRadius);


	float thetaIncr = M_PI / 90.0;

	for (int i = 0; i < mag.rows; i ++)
	{
		for (int j = 0; j < mag.cols; j ++)
		{
			if (mag(i,j) > threshold)
			{

			
				for (int r = 0; r < maxRadius; r ++)
				{
					float direction = dir(i, j);
					for (float theta = direction - thetaThreshold; theta < direction + thetaThreshold; theta += thetaIncr)
					{	
						int x0 = j + r * cosf(theta);
						int y0 = i + r * sinf(theta);

						int x1 = j - r * cosf(theta);
						int y1 = i - r * sinf(theta);

						if (x0 >= mag.cols || y0 >= mag.rows) continue;
						if (x0 < 0 || y0 < 0) continue;

						if (x1 >= mag.cols || y1 >= mag.rows) continue;
						if (x1 < 0 || y1 < 0) continue;

						H[y0][x0][r] ++;
						H[y1][x1][r] ++;
					}
				}
			}
		}
	}


	for (int i = 0; i < output.rows; i ++)
	{
		for (int j = 0; j < output.cols; j ++)
		{
			float sum = 0.0;
			for (int k = 0; k < maxRadius; k ++)
			{
				sum += H[i][j][k];
				
			}
			output(i, j) = sum;
		}
	}

}

void threshold(cv::Mat_<uchar> &input, cv::Mat_<uchar> &output, uchar t)
{
	output.create(input.size());
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			if (input(i, j) > t)
			{
				output(i, j) = 255;
			}
			else
			{
				output(i, j) = 0;
			}
		}
	}
}

int main()
{
	cv::Mat sourceColour = imread("img/coins1.png");
  	cv::Mat source;
  	cvtColor(sourceColour, source, CV_BGR2GRAY);


	cv::Mat_<float> houghOut;
	hough(source, houghOut, 20, 128, 0.05);


	cv::Mat_<float> logOut;
	cv::log(houghOut, logOut);


	cv::Mat_<uchar> rescaled;
	rescale(houghOut, rescaled);

	cv::Mat_<uchar> thresholdOut;
	threshold(rescaled, thresholdOut, 150);


	showResult(thresholdOut, "thresholded");
	convertAndShow(houghOut, "hough space");
	convertAndShow(logOut, "log hough space");

	return 0;
}