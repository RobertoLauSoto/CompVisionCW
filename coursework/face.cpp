/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

//define 2D array for bleh
int ground_truths[4][5];

void init_ground_truths() {
	ground_truths[0][0]=350;
	ground_truths[1][0]=104;
	ground_truths[2][0]=464;
	ground_truths[3][0]=265;
}

float get_iou(int a[4], int b[4]) {
	int x1 = max(a[0], b[0]);
	int y1 = max(a[1], b[1]);
	int x2 = min(a[2], b[2]);
	int y2 = min(a[3], b[3]);

	int width = x2 - x1;
	int height = y2 - y1;

	if(width<0 || height<0){
		return 0;
	}
	else {
		int area_overlap = width * height;

		int area_a = (a[2] - a[0])*(a[3] - a[1]);
		int area_b = (b[2] - b[0])*(b[3] - b[1]);
		int area_combined = area_a + area_b - area_overlap;
		float iou = (float)area_overlap / (float)area_combined;
		return iou;
	}
}

/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	init_ground_truths();

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		//Draw green
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
		//Draw red
		rectangle(frame, Point(ground_truths[0][0], ground_truths[1][0]), Point(ground_truths[2][0], ground_truths[3][0]), Scalar( 0, 0, 255 ), 2);
		
		int a[4] = {faces[i].x, faces[i].y, faces[i].x + faces[i].width, faces[i].y + faces[i].height};
		int b[4] = {ground_truths[0][0],ground_truths[1][0],ground_truths[2][0],ground_truths[3][0]};
		float iou = get_iou(a, b);
		std::cout << iou << std::endl;
	}
}


