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
void detectAndDisplay( Mat frame, int ino);

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

//define 2D array for bleh
int ground_truths[5][4][5];

void init_ground_truths() {
	//dart 4
	ground_truths[0][0][0]=350;
	ground_truths[0][1][0]=104;
	ground_truths[0][2][0]=464;
	ground_truths[0][3][0]=265;

	//dart5
	ground_truths[1][0][0]=65;
	ground_truths[1][1][0]=145;
	ground_truths[1][2][0]=122;
	ground_truths[1][3][0]=203;

	ground_truths[1][0][1]=268;
	ground_truths[1][1][1]=237;
	ground_truths[1][2][1]=327;
	ground_truths[1][3][1]=315;

	ground_truths[1][0][2]=407;
	ground_truths[1][1][2]=203;
	ground_truths[1][2][2]=459;
	ground_truths[1][3][2]=272;

	ground_truths[1][0][3]=463;
	ground_truths[1][1][3]=163;
	ground_truths[1][2][3]=516;
	ground_truths[1][3][3]=222;

	ground_truths[1][0][4]=508;
	ground_truths[1][1][4]=227;
	ground_truths[1][2][4]=554;
	ground_truths[1][3][4]=295;

	ground_truths[1][0][5]=585;
	ground_truths[1][1][5]=175;
	ground_truths[1][2][5]=650;
	ground_truths[1][3][5]=245;

	ground_truths[1][0][6]=642;
	ground_truths[1][1][6]=216;
	ground_truths[1][2][6]=695;
	ground_truths[1][3][6]=300;

	ground_truths[1][0][7]=726;
	ground_truths[1][1][7]=162;
	ground_truths[1][2][7]=790;
	ground_truths[1][3][7]=233;

	ground_truths[1][0][8]=774;
	ground_truths[1][1][8]=235;
	ground_truths[1][2][8]=827;
	ground_truths[1][3][8]=306;

	ground_truths[1][0][9]=861;
	ground_truths[1][1][9]=176;
	ground_truths[1][2][9]=916;
	ground_truths[1][3][9]=239;

	ground_truths[1][0][10]=895;
	ground_truths[1][1][10]=236;
	ground_truths[1][2][10]=942;
	ground_truths[1][3][10]=301;

	//dart13
	ground_truths[2][0][0]=416;
	ground_truths[2][1][0]=116;
	ground_truths[2][2][0]=527;
	ground_truths[2][3][0]=255;

	//dart14
	ground_truths[3][0][0]=470;
	ground_truths[3][1][0]=225;
	ground_truths[3][2][0]=546;
	ground_truths[3][3][0]=315;

	ground_truths[3][0][1]=730;
	ground_truths[3][1][1]=190;
	ground_truths[3][2][1]=827;
	ground_truths[3][3][1]=295;

	//dart15
	ground_truths[4][0][0]=58;
	ground_truths[4][1][0]=133;
	ground_truths[4][2][0]=120;
	ground_truths[4][3][0]=214;

	ground_truths[4][0][1]=367;
	ground_truths[4][1][1]=109;
	ground_truths[4][2][1]=442;
	ground_truths[4][3][1]=195;

	ground_truths[4][0][2]=539;
	ground_truths[4][1][2]=126;
	ground_truths[4][2][2]=621;
	ground_truths[4][3][2]=215;
}

int get_num_faces(int ino){
	int numface = 0;
	if (ino == 0) numface = 1;
	if (ino == 1) numface = 11;
	if (ino == 2) numface = 1; 
	if (ino == 3) numface = 2; 
	if (ino == 4) numface = 3;
	return numface; 
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

int get_image_number(string imagename){
	int imagenumber = 0;
	string digit1 = imagename.substr(4,1);
	if (digit1 == "4") imagenumber = 0;
	if (digit1 == "5") imagenumber = 1;
	if (digit1 == "1") {
		string digit2 = imagename.substr(5,1);
		if (digit2 == "3") imagenumber = 2;
		if (digit2 == "4") imagenumber = 3;
		if (digit2 == "5") imagenumber = 4;
	}
	return imagenumber;
}

/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	int image_number = get_image_number(argv[1]);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	init_ground_truths();

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame, image_number );

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, int ino )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << "Faces detected:" << std::endl;
	std::cout << faces.size() << std::endl;
	std::cout << "Actual number of faces:" << std::endl;
	std::cout << get_num_faces(ino) << std::endl;

       // 4. Draw box around faces found
	for( int di = 0; di < faces.size(); di++ ){
		//Draw green
		rectangle(frame, Point(faces[di].x, faces[di].y), Point(faces[di].x + faces[di].width, faces[di].y + faces[di].height), Scalar( 0, 255, 0 ), 2);
	}

	//Draw red
	for( int dj = 0; dj < get_num_faces(ino); dj++ ){	
		rectangle(frame, Point(ground_truths[ino][0][dj], ground_truths[ino][1][dj]), Point(ground_truths[ino][2][dj], ground_truths[ino][3][dj]), Scalar( 0, 0, 255 ), 2);
	}

	//Compare
	for( int i = 0; i < faces.size(); i++ ){
		float best_iou = 0;
		for ( int j = 0; j < get_num_faces(ino); j++ ){
			int a[4] = {faces[i].x, faces[i].y, faces[i].x + faces[i].width, faces[i].y + faces[i].height};
			int b[4] = {ground_truths[ino][0][j],ground_truths[ino][1][j],ground_truths[ino][2][j],ground_truths[ino][3][j]};
			float iou = get_iou(a, b);
			if (iou > best_iou) best_iou = iou;
		}
		std::cout << best_iou << std::endl;
	}
}


