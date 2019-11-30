#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include "sobel.h"
#include "3dArray.h"
#include "convolution.h"
#include <cmath>

using namespace std;
using namespace cv;

String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

//define 2D array for ground truths
int ground_truths[16][4][3];

//Init ground truth vaules for dartboards
void init_ground_truths(){
	//Top left and top right
	//Dart0
	ground_truths[0][0][0] = 426;
	ground_truths[0][1][0] = 4;
	ground_truths[0][2][0] = 612;
	ground_truths[0][3][0] = 217;
	//Dart1
	ground_truths[1][0][0] = 166;
	ground_truths[1][1][0] = 99;
	ground_truths[1][2][0] = 424;
	ground_truths[1][3][0] = 355;
	//Dart2
	ground_truths[2][0][0] = 93;
	ground_truths[2][1][0] = 92;
	ground_truths[2][2][0] = 205;
	ground_truths[2][3][0] = 194;
	//Dart3
	ground_truths[3][0][0] = 314;
	ground_truths[3][1][0] = 149;
	ground_truths[3][2][0] = 397;
	ground_truths[3][3][0] = 228;
	//Dart4
	ground_truths[4][0][0] = 156;
	ground_truths[4][1][0] = 71;
	ground_truths[4][2][0] = 353;
	ground_truths[4][3][0] = 305;
	//Dart5
	ground_truths[5][0][0] = 423;
	ground_truths[5][1][0] = 141;
	ground_truths[5][2][0] = 532;
	ground_truths[5][3][0] = 240;
	//Dart6
	ground_truths[6][0][0] = 202;
	ground_truths[6][1][0] = 108;
	ground_truths[6][2][0] = 278;
	ground_truths[6][3][0] = 187;
	//Dart7
	ground_truths[7][0][0] = 238;
	ground_truths[7][1][0] = 158;
	ground_truths[7][2][0] = 383;
	ground_truths[7][3][0] = 329;
	//Dart8
	ground_truths[8][0][0] = 64;
	ground_truths[8][1][0] = 258;
	ground_truths[8][2][0] = 126;
	ground_truths[8][3][0] = 350;
	ground_truths[8][0][1] = 832;
	ground_truths[8][1][1] = 212;
	ground_truths[8][2][1] = 974;
	ground_truths[8][3][1] = 345;
	//Dart9
	ground_truths[9][0][0] = 180;
	ground_truths[9][1][0] = 31;
	ground_truths[9][2][0] = 460;
	ground_truths[9][3][0] = 296;
	//Dart10
	ground_truths[10][0][0] = 79;
	ground_truths[10][1][0] = 99;
	ground_truths[10][2][0] = 199;
	ground_truths[10][3][0] = 224;
	ground_truths[10][0][1] = 573;
	ground_truths[10][1][1] = 126;
	ground_truths[10][2][1] = 651;
	ground_truths[10][3][1] = 221;
	ground_truths[10][0][2] = 914;
	ground_truths[10][1][2] = 146;
	ground_truths[10][2][2] = 951;
	ground_truths[10][3][2] = 219;
	//Dart11
	ground_truths[11][0][0] = 167;
	ground_truths[11][1][0] = 102;
	ground_truths[11][2][0] = 240;
	ground_truths[11][3][0] = 156;
	ground_truths[11][0][1] = 434;
	ground_truths[11][1][1] = 122;
	ground_truths[11][2][1] = 460;
	ground_truths[11][3][1] = 183;
	//Dart12
	ground_truths[12][0][0] = 151;
	ground_truths[12][1][0] = 67;
	ground_truths[12][2][0] = 225;
	ground_truths[12][3][0] = 237;
	//Dart13
	ground_truths[13][0][0] = 259;
	ground_truths[13][1][0] = 111;
	ground_truths[13][2][0] = 414;
	ground_truths[13][3][0] = 256;
	//Dart14
	ground_truths[14][0][0] = 99;
	ground_truths[14][1][0] = 94;
	ground_truths[14][2][0] = 260;
	ground_truths[14][3][0] = 242;
	ground_truths[14][0][1] = 975;
	ground_truths[14][1][1] = 75;
	ground_truths[14][2][1] = 1124;
	ground_truths[14][3][1] = 240;
	//Dart15
	ground_truths[15][0][0] = 142;
	ground_truths[15][1][0] = 40;
	ground_truths[15][2][0] = 298;
	ground_truths[15][3][0] = 213;
}

//For what image get number of dartboards
int get_num_actual_dartboards(int ino){
	int dartnum = 0;
	if (ino == 0) dartnum = 1;
	if (ino == 1) dartnum = 1;
	if (ino == 2) dartnum = 1; 
	if (ino == 3) dartnum = 1; 
	if (ino == 4) dartnum = 1;
	if (ino == 5) dartnum = 1;
	if (ino == 6) dartnum = 1;
	if (ino == 7) dartnum = 1;
	if (ino == 8) dartnum = 2;
	if (ino == 9) dartnum = 1;
	if (ino == 10) dartnum = 3;
	if (ino == 11) dartnum = 2;
	if (ino == 12) dartnum = 1;
	if (ino == 13) dartnum = 1;
	if (ino == 14) dartnum = 2;
	if (ino == 15) dartnum = 1;
	return dartnum; 
}

//Manual input for f1score - done after detector was finished
int get_num_detected_dartboards(int ino){
	int detected = 0;
	if (ino == 0) detected = 1;
	if (ino == 1) detected = 1;
	if (ino == 2) detected = 1; 
	if (ino == 3) detected = 0; 
	if (ino == 4) detected = 1;
	if (ino == 5) detected = 1;
	if (ino == 6) detected = 0;
	if (ino == 7) detected = 1;
	if (ino == 8) detected = 1;
	if (ino == 9) detected = 1;
	if (ino == 10) detected = 2;
	if (ino == 11) detected = 0;
	if (ino == 12) detected = 0;
	if (ino == 13) detected = 1;
	if (ino == 14) detected = 2;
	if (ino == 15) detected = 1;

	return detected;

}

//From the given file get the image number
int get_image_number(string imagename){
	int imagenumber = 0;
	string digit1 = imagename.substr(4,1);
	string digit2 = imagename.substr(5,1);
	if (digit2 == ".") {
		if (digit1 == "0") imagenumber = 0;
		if (digit1 == "1") imagenumber = 1;
		if (digit1 == "2") imagenumber = 2;
		if (digit1 == "3") imagenumber = 3;
		if (digit1 == "4") imagenumber = 4;
		if (digit1 == "5") imagenumber = 5;
		if (digit1 == "6") imagenumber = 6;
		if (digit1 == "7") imagenumber = 7;
		if (digit1 == "8") imagenumber = 8;
		if (digit1 == "9") imagenumber = 9;
	} else if (digit1 == "1" && digit2 != "."){
		if (digit2 == "0") imagenumber = 10;
		if (digit2 == "1") imagenumber = 11;
		if (digit2 == "2") imagenumber = 12;
		if (digit2 == "3") imagenumber = 13;
		if (digit2 == "4") imagenumber = 14;
		if (digit2 == "5") imagenumber = 15;
	}
	
	return imagenumber;
}

//Calculate the f1 score
float calculate_f1_score (float precision, float recall) {
	float f1_score = 2 * ((precision * recall) / (precision + recall));
	return f1_score;
}

//Calculate the intersection over union for two rectangles
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

//Check if a line goes through a circle
bool circle_line_intersect(int cx, int cy, int r, int x1, int y1, int x2, int y2){
	bool intersect = false;
	float num = (x2 - x1) * cx + (y1 - y2) * cy + (x1 - x2) * y1 + (y2 - y1) * x1;
	float denom = sqrt( (x2 - x1)*(x2 - x1) + (y1 - y2)*(y1 - y2) );
	if (num < 0) num = num - (2*num);
	if ((num / denom) > r) intersect = 1; 

	return intersect;
}

//Output hough spaces
void output_hough_circle(int*** HC, const cv::Mat_<uchar> &image, const int rmin, const int rmax, cv::Mat_<float> &mag){
	cv::Mat_<float> output = mag.clone();
	cv::Mat_<float> temp = mag.clone();
	float max = 0.0;
	float min = 1000.0;
	for (int i = 0; i < mag.cols; i++){
		for (int j = 0; j < mag.rows; j++){
			float total = 0.0;
			for (int r = rmin; r < rmax; r++){
				total += (float)HC[j][i][r];
			}
			if (total > max) max = total;
			if (total < min) min = total;
			temp(j,i) = total;
		}
	}
	for (int i = 0; i < mag.cols; i++){
		for (int j = 0; j < mag.rows; j++){
			output(j,i) = min + temp(j,i) * 255.0 / max;
		}
	}

	imwrite("houghcircle.jpg", output);
}

//Draw circles from the circle hough
void draw_circles(int*** H, int rmin, int rmax, int thresh_acc, Mat& originalimage){
	for (int y = 0; y < originalimage.rows; y ++){
		for (int x = 0; x < originalimage.cols; x ++){
			for (int r = rmin; r < rmax; r ++){
				int acc = H[y][x][r];
				if (acc > thresh_acc){
					circle(originalimage, Point(x, y), r, cvScalar(3,140,252), 1);
				}
			}
		}
	}
}

//Draw lines from the line hough
void draw_lines(int** HL, int thresh_acc, int theta_numsteps, int rhomax, Mat& originalimage){
	float trads = (2*CV_PI) / (float)theta_numsteps;
	for (int rho = 0; rho < rhomax; rho++){
		float radcount = 0;
		for (int theta = 0; theta < theta_numsteps; theta++){
			int acc = HL[rho][theta];
			if (acc > thresh_acc){
				int x = rho * cosf(radcount);
				int y = rho * sinf(radcount);
				
				int x1 = x - y * 100;
				int x2 = x + y * 100;
				int y1 = y + x * 100;
				int y2 = y - x * 100;

				line(originalimage, Point(x1, y1), Point(x2, y2), Scalar(255,0,123), 1);
			}
			radcount += trads;
		}
	}
}

//GO through hough lines, find out if they intersect circle 
int hough_lines(int** HL, int thresh_acc, int theta_numsteps, int rhomax, Mat& originalimage, bool drawhough, int bestcentre[3]){
	int totallines = 0;
	float trads = (2*CV_PI) / (float)theta_numsteps;
	for (int rho = 0; rho < rhomax; rho++){
		float radcount = 0;
		for (int theta = 0; theta < theta_numsteps; theta++){
			int acc = HL[rho][theta];
			if (acc > thresh_acc){
				int x = rho * cosf(radcount);
				int y = rho * sinf(radcount);
				
				int x1 = x - y * 20;
				int x2 = x + y * 20;
				int y1 = y + x * 20;
				int y2 = y - x * 20;

				if (circle_line_intersect(bestcentre[1], bestcentre[2], 20, x1, y1, x2, y2) == 1) totallines++;
				if (drawhough) line(originalimage, Point(x1, y1), Point(x2, y2), Scalar(255,0,123), 1);
			}
			radcount += trads;
		}
	}
	return totallines;
}

//Line hough transform
int** hough_line_transform(std::vector<Rect> dartboards, int db, const cv::Mat_<uchar> &image, const uchar thresh_magnitude, const float thresh_theta, const int theta_numsteps, const int rhomax){
	cv::Mat_<float> dx;
	cv::Mat_<float> dy;
	cv::Mat_<float> mag;
	cv::Mat_<float> dir;
	sobel(image, dx, dy, mag, dir);

	//Allocate array
	int **HL = malloc2dArray(rhomax, theta_numsteps);

	//Convert to radians
	float theta_incr = (2*CV_PI) / (float)theta_numsteps;

	for (int i = dartboards[db].y; i < dartboards[db].y + dartboards[db].height; i++){
		for (int j = dartboards[db].x; j < dartboards[db].x + dartboards[db].width; j++){
			if (mag(i, j) > thresh_magnitude){
				float direction = dir(i, j);
				for (float theta = direction - thresh_theta; theta < direction + thresh_theta; theta += theta_incr){
					int rho = j * cosf(theta) + i * sinf(theta);
					if (rho < 0 || rho >= rhomax) continue;
					int thetaDeg = theta / theta_incr;
					if (thetaDeg < 0) thetaDeg += theta_numsteps;
					if (thetaDeg > theta_numsteps) thetaDeg -= theta_numsteps;

					HL[rho][thetaDeg]++;
				}
			}
		}
	}

	return HL;
}

//Circle hough transform
int*** hough_circle_transform(const cv::Mat_<uchar> &image, const uchar thresh_magnitude, const int rmin, const int rmax, const float thetaThreshold, const float thetaIncrCoeff){
	cv::Mat_<float> dx;
	cv::Mat_<float> dy;
	cv::Mat_<float> mag;
	cv::Mat_<float> dir;
	
    sobel(image, dx, dy, mag, dir);

	//Draw magnitude image
	cv::Mat_<uchar> magout = mag.clone();
	for (int i = 0; i < mag.rows; i ++){
		for (int j = 0; j < mag.cols; j ++){
			if (mag(i,j) > thresh_magnitude){
				magout(i,j) = 255;
			} else {
				magout(i,j) = 0;
			}
		}
	}
	imwrite("magnitude.jpg", magout);

	int ***HC = malloc3dArray(image.rows, image.cols, rmax);
    float theta_incr = (2*CV_PI) / thetaIncrCoeff;

	for (int i = 0; i < mag.rows; i ++){
		for (int j = 0; j < mag.cols; j ++){
			if (mag(i,j) > thresh_magnitude){
				for (int r = rmin; r < rmax; r++){
					float direction = dir(i, j);
					for (float theta = direction - thetaThreshold; theta < direction + thetaThreshold; theta += theta_incr){	
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

	output_hough_circle(HC, image, rmin, rmax, mag);

	return HC;
}

int main( int argc, const char** argv ){
    // 1. Read Input Image
	cv::Mat colourimage = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat_<uchar> image;
    cvtColor(colourimage, image, CV_BGR2GRAY);
	std::vector<Rect> dartboards;
	std::vector<Rect> detections;
	std::vector<float> confidences;
	init_ground_truths();
	int image_number = get_image_number(argv[1]);
	int numdbpass = 0;

	//-------------------------
	//Variables and thresholds
	//-------------------------
	//True if want to draw extra data onto the image
	bool drawvj = true;
	bool drawhough = true;
	bool drawunaveraged = true;

	//Radius max and min
	int rmin = 25;
    int rmax = 150;

	//Magnitude threshold
	int thresh_magnitude = 80;

	//Accumulator threshold for circles
	int thresh_acc_cirles = 100;

	//Accumulator threshold for lines
	int thresh_acc_lines = 25;

	//Number of circles +1 with same centre threshold
	int thresh_samecentre = 2;

	//Threshold for number of centre of circles over number of circles threshold
	int thresh_numcentres = 20;

	// Threshold for number of lines going through the centre of a dartboard
	int thresh_numlines = 100; 

	//Number of steps theta goes through per 2 pi for houghlines
	int theta_numsteps = 720;

	//Highest value of rho in houghlines
	int rhomax = 700;

	//Threshold for iou to tell if two vj boxes look at the same dartboard
	int thresh_overlap = 0.25;


	//------------
	//ViolaJones
	//------------
	//Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	float vjthreshold = 0.25;
	cascade.detectMultiScale(image, dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
	
	//Draw viola jones detected dartboards
	if (drawvj){
		for(int di = 0; di < dartboards.size(); di++){
			rectangle(colourimage, Point(dartboards[di].x, dartboards[di].y), Point(dartboards[di].x + dartboards[di].width, dartboards[di].y + dartboards[di].height), Scalar( 20, 255, 20 ), 1);
		}
	}
	//------------

	//-----------------
	// Hough Circle Transform
	//-----------------
    int ***HC = hough_circle_transform(image, thresh_magnitude, rmin, rmax, 0.1, 360.0);
	//Draw Hough Circle Output
	if (drawhough) draw_circles(HC, rmin, rmax, thresh_acc_cirles, colourimage);
	//-----------------


	//------------------------------
	//Circle centre scoring
	//------------------------------
	int numcircles[colourimage.rows][colourimage.cols];
	//Init
	for (int y = 0; y < colourimage.rows; y ++){
		for (int x = 0; x < colourimage.cols; x ++){
			numcircles[y][x] = 0;
		}
	}

	//Circle Centre basic scoring
	for (int y = 0; y < colourimage.rows; y ++){
		for (int x = 0; x < colourimage.cols; x ++){
			for (int r = rmin; r < rmax; r ++){
				int votes = HC[y][x][r];
				if (votes > thresh_acc_cirles){
					numcircles[y][x]++;
				}
			}
		}
	}
	//------------------------------


	//--------------------------------
	//Evaluation
	//--------------------------------
	//Loop over Viola Jones areas
	for(int db = 0; db < dartboards.size(); db++){
		//score for each "green square" / potential vj detected dartboard
		int dbcentrescore = 0;
		for (int y = 0; y < colourimage.rows; y ++){
			for (int x = 0; x < colourimage.cols; x ++){
				//If there are more than a threshold number of circle centres at the point
				if (numcircles[y][x] > thresh_samecentre){
					//check if point is within vj area
					if (dartboards[db].x < x && x < dartboards[db].x + dartboards[db].width) {
						if (dartboards[db].y < y && y < dartboards[db].y + dartboards[db].height) {
							//Point IS within area
							//check for more centres nearby
							for (int y2 = y-1; y2 < y+1; y2++){
								for (int x2 = x-1; x2 < x+1; x2++){
									if (numcircles[y2][x2] != 0) numcircles[y][x]++;
								}
							}

							if (drawhough) circle(colourimage, Point(x, y), 1, cvScalar(255,0,0), 1);

							dbcentrescore = dbcentrescore + numcircles[y][x];
						} 	
					} 
				}
			}
		}

		//Find best centre point in area
		int bestcentre[3] = {0,0,0};
		for (int x = dartboards[db].x; x < dartboards[db].x + dartboards[db].width; x++){
			for (int y = dartboards[db].y; y < dartboards[db].y + dartboards[db].height; y++){
				if (numcircles[y][x] > bestcentre[0]){
					bestcentre[0] = numcircles[y][x];
					bestcentre[1] = x;
					bestcentre[2] = y;
				}
			}
		}


		//---------------------
		//Hough Line Transform
		//---------------------
		int **HL = hough_line_transform(dartboards, db, image, thresh_magnitude, 0.1, theta_numsteps, rhomax);
		//Find out how many lines cross the centre
		int linescore = hough_lines(HL, thresh_acc_lines, theta_numsteps, rhomax, colourimage, drawhough, bestcentre);
		//---------------------


		//----------------------
		//Evaluate and Draw Bounding Boxes
		//----------------------
		//If the number of centres of multiple circles is over the threshold and is in a Viola Jones area
		// and the the hough lines is over a threshold then combine the two and detect a dartboard
		// Output a confindence
		//This combines both hough transforms and viola jones
		if (linescore > thresh_numlines){
			if (drawhough) rectangle(colourimage, Point(dartboards[db].x, dartboards[db].y), Point(dartboards[db].x + dartboards[db].width, dartboards[db].y + dartboards[db].height), Scalar( 255, 0, 0 ), 3);
		}
		if (dbcentrescore > thresh_numcentres){
			float confidence = 0.7;
			confidence = confidence + ((float)dbcentrescore / 1000) + ((float)linescore / 1000) ;
			if (confidence > 1) confidence = 1;
			int px = bestcentre[1] - (dartboards[db].width / 2);
			int py = bestcentre[2] - (dartboards[db].height / 2);
				
			Rect box = {px, py, dartboards[db].width, dartboards[db].height};
			detections.push_back(box);
			confidences.push_back(confidence);
			if (drawunaveraged) rectangle(colourimage, Point(px, py), Point(px + dartboards[db].width, py + dartboards[db].height), Scalar( 0, 255, 255 ), 2);
		}
		//----------------------
	}
	//--------------------------------


	//-------------------
	//Draw ground truths
	//-------------------
	for( int dj = 0; dj < get_num_actual_dartboards(image_number); dj++ ){	
		rectangle(colourimage, Point(ground_truths[image_number][0][dj], ground_truths[image_number][1][dj]), Point(ground_truths[image_number][2][dj], ground_truths[image_number][3][dj]), Scalar( 0, 0, 255 ), 2);
	}

	
	//-------------------
	//Final Evaluation
	//-------------------
	//If there are many detections of the same dartboard fix that here by making an average distance
	// based on weighted confidence
	
	//Get iou between dartboards
	float dartboardsious[detections.size()][detections.size()];
	for (int db = 0; db < detections.size(); db++){
		for (int db2 = 0; db2 < detections.size(); db2++){
			int a[4] = {detections[db].x, detections[db].y, detections[db].x + detections[db].width, detections[db].y + detections[db].height};
			int b[4] = {detections[db2].x, detections[db2].y, detections[db2].x + detections[db2].width, detections[db2].y + detections[db2].height};
			float iou = get_iou(a, b);
			if (db == db2){
				dartboardsious[db][db2] = 0;
			} else {
				dartboardsious[db][db2] = iou;
			}	
		}
	}
	
	//Find overlapping dartboards
	std::vector<int> overlaps;
	for (int db = 0; db < detections.size(); db++){
		for (int db2 = 0; db2 < detections.size(); db2++){
			if (dartboardsious[db][db2] > thresh_overlap) overlaps.push_back(db2);
		} 
	}

	sort(overlaps.begin(),overlaps.end());
	overlaps.erase( unique(overlaps.begin(),overlaps.end() ),overlaps.end());

	//Average output
	if (!overlaps.empty()){
		float avgx = 0;
		float avgy = 0;
		float avgwidth = 0;
		float avgheight = 0;
		for (int i = 0; i < overlaps.size(); i ++){
			avgx += detections[overlaps[i]].x;
			avgy += detections[overlaps[i]].y;
			avgwidth += detections[overlaps[i]].width;
			avgheight += detections[overlaps[i]].height;
		}
		avgx = avgx / overlaps.size();
		avgy = avgy / overlaps.size();
		avgwidth = avgwidth / overlaps.size();
		avgheight = avgheight / overlaps.size();
		rectangle(colourimage, Point(avgx, avgy), Point(avgx + avgwidth, avgy + avgheight), Scalar( 0, 255, 0 ), 3);
		std::cout << "Dartboard Detected with confidence:" << std::endl;
		std::cout << confidences[overlaps.front()] << std::endl;
		numdbpass++;
	}
	//-------------------


	//-------------------------------------
	//Output number of detected dartboards
	//-------------------------------------
	for (int i = 0; i < detections.size(); i++){
		if (!(std::find(overlaps.begin(), overlaps.end(), i) != overlaps.end())){
			rectangle(colourimage, Point(detections[i].x, detections[i].y), Point(detections[i].x + detections[i].width, detections[i].y + detections[i].height), Scalar( 0, 255, 0 ), 3);
			std::cout << "Dartboard Detected with confidence:" << std::endl;
			std::cout << confidences[i] << std::endl;
			numdbpass++;
		} 
	}

	std::cout << "Number of dartboards passing all detectors:" << std::endl;
	std::cout << numdbpass << std::endl;
	std::cout << "Out of:" << std::endl;
	std::cout << get_num_actual_dartboards(image_number) << std::endl;
	//-------------------------------------

	/*
	//Print f1 score
	float precision = ((float) get_num_detected_dartboards(image_number)/ (float) dartboards.size());
	float recall = ((float) get_num_detected_dartboards(image_number) / (float) get_num_actual_dartboards(image_number));
	float f1_score = calculate_f1_score(precision, recall);
	std::cout << "F1 score:" << std::endl;
	std::cout << f1_score << std::endl;
	*/
	
    imwrite("detected.jpg", colourimage );
    return 0;
}