#include <opencv/cv.h>

void convolution(const cv::Mat_<uchar> &input, cv::Mat_<float> &kernel, cv::Mat_<float> &output);

void sobel(const cv::Mat_<uchar> &input, cv::Mat_<float> &dxOut, cv::Mat_<float> &dyOut, cv::Mat_<float> &magOut, cv::Mat_<float> &dirOut);

