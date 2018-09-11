#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;


#define window_width 640
#define window_height 480

vector<int> CalculateXYPixel(Rect location);