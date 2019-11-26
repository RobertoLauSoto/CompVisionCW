#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include "sobel.h"
#include "3dArray.h"
#include "convolution.h"

using namespace std;
using namespace cv;

String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

void threshold_image(cv::Mat_<uchar> &image, cv::Mat_<uchar> &output, uchar t){
	output.create(image.size());
	for (int i = 0; i < image.rows; i++){
		for (int j = 0; j < image.cols; j++){
			if (image(i, j) > t){
				output(i, j) = 255;
			} else {
				output(i, j) = 0;
			}
		}
	}
}

void draw_circles(int*** H, int rmin, int rmax, int threshold, Mat& originalimage){
	for (int y = 0; y < originalimage.rows; y ++){
		for (int x = 0; x < originalimage.cols; x ++){
			for (int r = rmin; r < rmax; r ++){
				int votes = H[y][x][r];

				if (votes > threshold){
					circle(originalimage, Point(x, y), r, cvScalar(0,0,255), 1);
				}
			}
		}
	}
}

void draw_lines(vector<Vec2f> HL, int threshold, Mat& originalimage){
	for(size_t i = 0; i < HL.size(); i++){
  		float rho = HL[i][0], theta = HL[i][1];
  		Point pt1, pt2;
  		double a = cos(theta), b = sin(theta);
  		double x0 = a*rho, y0 = b*rho;
  		pt1.x = cvRound(x0 + 1000*(-b));
  		pt1.y = cvRound(y0 + 1000*(a));
  		pt2.x = cvRound(x0 - 1000*(-b));
  		pt2.y = cvRound(y0 - 1000*(a));
  		line( originalimage, pt1, pt2, Scalar(0,0,255), 1, CV_AA);
	}
}

int** hough_line_transform(const cv::Mat_<uchar> &image, const uchar mthreshold, const float thetaThreshold, const float thetaIncrCoeff){
	cv::Mat_<float> dx;
	cv::Mat_<float> dy;
	cv::Mat_<float> mag;
	cv::Mat_<float> dir;
	
    sobel(image, dx, dy, mag, dir);
	float thetaIncr = (2*CV_PI) / thetaIncrCoeff;
	int max = mag.rows + mag.cols + 1;
	int **HL = malloc2dArray(max, 360);
	
	for (int i = 0; i < mag.rows; i++){
		for (int j = 0; j < mag.cols; j ++){
			//Magnitude threshold
			if (mag(i,j) > mthreshold){
				float direction = dir(i, j);
				int r = i*cosf(direction) + j * sinf(direction);

				HL[1][int(direction)]++;
			}
		}
	}

	return HL;
}


int*** hough_circle_transform(const cv::Mat_<uchar> &image, const uchar mthreshold, const int rmin, const int rmax, const float thetaThreshold, const float thetaIncrCoeff){
	cv::Mat_<float> dx;
	cv::Mat_<float> dy;
	cv::Mat_<float> mag;
	cv::Mat_<float> dir;
	
    sobel(image, dx, dy, mag, dir);
	int ***HC = malloc3dArray(image.rows, image.cols, rmax);
    float thetaIncr = (2*CV_PI) / thetaIncrCoeff;

	for (int i = 0; i < mag.rows; i ++){
		for (int j = 0; j < mag.cols; j ++){
			if (mag(i,j) > mthreshold){
				for (int r = rmin; r < rmax; r++){
					float direction = dir(i, j);
					for (float theta = direction - thetaThreshold; theta < direction + thetaThreshold; theta += thetaIncr){	
						for (int a = -1; a <= 1; a +=2){
							for (int b = -1; b <= 1; b +=2){
								int x0 = j + (a*r) * cosf(theta);
								int y0 = i + (b*r) * sinf(theta);
								if (x0 >= mag.cols || y0 >= mag.rows) continue;
								if (x0 < 0 || y0 < 0) continue;

								HC[y0][x0][r] ++;
							}
						}
					}
				}
			}
		}
	}

	return HC;
}

void violajones(const cv::Mat_<uchar> &image){

}

int main( int argc, const char** argv ){
    // 1. Read Input Image
	cv::Mat colourimage = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat_<uchar> image;
    cvtColor(colourimage, image, CV_BGR2GRAY);
	std::vector<Rect> dartboards;

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    int rmin = 10;
    int rmax = 150;
    int mthreshold = 80;
    int threshold = 100;

	// Hough
    int ***HC = hough_circle_transform(image, mthreshold, rmin, rmax, 0.1, 360.0);
	//int **HL = hough_line_transform(image, mthreshold, 0.1, 360);

	draw_circles(HC, rmin, rmax, threshold, colourimage);
	//draw_lines(HL, threshold, colourimage);

	//ViolaJones
	cascade.detectMultiScale(image, dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
	std::cout << "Dartboards detected:" << std::endl;
	std::cout << dartboards.size() << std::endl;

	//Draw dartboards
	for( int di = 0; di < dartboards.size(); di++ ){
		rectangle(colourimage, Point(dartboards[di].x, dartboards[di].y), Point(dartboards[di].x + dartboards[di].width, dartboards[di].y + dartboards[di].height), Scalar( 0, 255, 0 ), 2);
	}


    imwrite( "houghoutput.jpg", colourimage );
    return 0;
}