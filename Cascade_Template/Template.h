#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class Template {
public:
	Template();
	void initTracking(Mat frame, Rect box, int scale = 2);
	Rect track(Mat frame);
	Rect getLocation();
	void limitRect(Rect &location, Size sz);
private:
	Mat model;
	Rect location;
	int scale;
};

