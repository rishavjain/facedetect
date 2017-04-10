#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

const string FACE_CASCADE = "haarcascade_frontalface_default.xml";
const string EYE_CASCADE = "haarcascade_eye.xml";

cv::CascadeClassifier loadClassifier(string filename);
cv::Mat loadImage(string filename);

int main(int argc, char** argv) {
	string input_name;
	
	// get the arguments
	if( argc == 2 ) {
		input_name = argv[1];
	} else {
		cerr << "Program requires one input file name." << endl;
		exit(0);
	}
		
	// load the face and eye classifiers
	cv::CascadeClassifier face_cascade = loadClassifier(FACE_CASCADE);
	cv::CascadeClassifier eye_cascade = loadClassifier(EYE_CASCADE);
	
	// load the input image from file
	cv::Mat img = loadImage(input_name);

	// convert image to grayscale
	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	cv::imshow("Grayscale Image", img_gray);
	
	// end program
	cv::waitKey(0);
	cout << endl;
	return 0;
}


cv::CascadeClassifier loadClassifier(string filename) {
	cv::CascadeClassifier cascade;
	
	if( !cascade.load(filename) ) {
		cerr << "Unable to load cascade from file: " << filename << endl;
		exit(0);
	}
	
	cout << "Cascade loaded from file: " << filename << endl;
	
	return cascade;
}

cv::Mat loadImage(string filename) {
	cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
	
	if( !img.data ) {
		cerr << "Unable to load image from file: " << filename << endl;
		exit(0);
	}
	
	cout << "Image loaded from file: " << filename << endl;
	cv::imshow("Input Image", img);
	
	return img;
}
