#include "ImageProcessing.h"



bool JudgeRGB(Mat src)
{
	int blue_count = 0;
	int red_count = 0;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<Vec3b>(i, j)[0] > 17 && src.at<Vec3b>(i, j)[0] < 50 &&
				src.at<Vec3b>(i, j)[1] > 15 && src.at<Vec3b>(i, j)[1] < 56 &&
				src.at<Vec3b>(i, j)[2] > 100 && src.at<Vec3b>(i, j)[2] < 250)
				red_count++;
			else if (
				src.at<Vec3b>(i, j)[0] > 86 && src.at<Vec3b>(i, j)[0] < 220 &&
				src.at<Vec3b>(i, j)[1] > 31 && src.at<Vec3b>(i, j)[1] < 88 &&
				src.at<Vec3b>(i, j)[2] > 4 && src.at<Vec3b>(i, j)[2] < 50)
				blue_count++;
		}
	}
	cout << "[debug] " << "blue_count: " << blue_count << "\tred_count: " << red_count << endl;
	if ((red_count - blue_count) >= 50)
		return true;
	else
		return false;
}

//HSV颜色空间颜色判别，注意因为共享内存地址，判别后需要返回BGR空间，否则影响追踪模板。
bool JudgeHSV(Mat src)
{
	int blue_count = 0;
	int red_count = 0;
	cvtColor(src, src, CV_BGR2HSV);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (((src.at<Vec3b>(i, j)[0] >= 0 && src.at<Vec3b>(i, j)[0] < 15) || (src.at<Vec3b>(i, j)[0] > 155 && src.at<Vec3b>(i, j)[0] <= 255)) &&
				src.at<Vec3b>(i, j)[1] > 50 && src.at<Vec3b>(i, j)[1] <= 255 &&
				src.at<Vec3b>(i, j)[2] > 50 && src.at<Vec3b>(i, j)[2] <= 255)
				red_count++;
			else if (
				src.at<Vec3b>(i, j)[0] > 95 && src.at<Vec3b>(i, j)[0] < 130 &&
				src.at<Vec3b>(i, j)[1] > 50 && src.at<Vec3b>(i, j)[1] <= 255 &&
				src.at<Vec3b>(i, j)[2] > 50 && src.at<Vec3b>(i, j)[2] <= 255)
				blue_count++;
		}
	}
	cvtColor(src, src, CV_HSV2BGR);
	cout << "[debug] " << "blue_count: " << blue_count << "\tred_count: " << red_count << endl;
	if ((red_count - blue_count) >= 50)
		return true;
	else
		return false;
}

vector<Rect> ColorFilter(Mat frame, vector<Rect> boards, bool color_flag)	//color filter
{
	vector<Rect> results;
	for (int i = 0; i < boards.size(); i++)
	{
		Mat roi = frame(boards[i]);
		//imshow("roi", roi);
		//waitKey(0);
		bool flag = JudgeRGB(roi);
		if (flag == color_flag)
			results.push_back(boards[i]);
	}
	//cout << results.size() << endl;
	return results;
}

void SaveTarget(Rect location, int frame_num, Mat frame)
{
	Mat frame1 = frame;
	rectangle(frame1, cvPoint(cvRound(location.x), cvRound(location.y)),
		cvPoint(cvRound((location.x + location.width - 1)), cvRound((location.y + location.height - 1))),
		Scalar(0, 255, 255), 3, 8, 0);
	char buf[30];
	sprintf_s(buf, "D:\\frame\\%d.jpg", frame_num);
	imwrite(buf, frame1);
}

void SaveAllBoard(vector<Rect> boards, int frame_num, Mat frame)
{
	Mat frame1 = frame;
	for (int i = 0;i < boards.size();i++)
	{
		rectangle(frame1, cvPoint(cvRound(boards[i].x), cvRound(boards[i].y)),
			cvPoint(cvRound((boards[i].x + boards[i].width - 1)), cvRound((boards[i].y + boards[i].height - 1))),
			Scalar(0, 255, 255), 3, 8, 0);
	}
	char buf[30];
	sprintf_s(buf, "D:\\frame3\\%d.jpg", frame_num);
	imwrite(buf, frame1);
}