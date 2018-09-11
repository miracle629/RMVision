#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

bool JudgeRGB(Mat src);
bool JudgeHSV(Mat src);
vector<Rect> ColorFilter(Mat frame, vector<Rect> boards, bool color_flag);	//color filter
void SaveTarget(Rect location, int frame_num, Mat frame);
void SaveAllBoard(vector<Rect> boards, int frame_num, Mat frame);