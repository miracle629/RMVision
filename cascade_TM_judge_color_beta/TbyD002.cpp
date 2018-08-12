#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

#define RED true
#define BLUE false

class Template {
public:
	Template();
	void initTracking(Mat frame, Rect box, int scale = 5);
	Rect track(Mat frame);
	Rect getLocation();
private:
	Mat model;
	Rect location;
	int scale;
};

Template::Template()
{
	this->scale = 3;
}

void Template::initTracking(Mat frame, Rect box, int scale)
{
	this->location = box;
	this->scale = scale;

	if (frame.empty())
	{
		cout << "ERROR: frame is empty." << endl;
		exit(0);
	}
	if (frame.channels() != 1)
	{
		cvtColor(frame, frame, CV_RGB2GRAY);
	}
	this->model = frame(box);
}

Rect Template::track(Mat frame)
{
	if (frame.empty())
	{
		cout << "ERROR: frame is empty." << endl;
		exit(0);
	}
	Mat gray;
	if (frame.channels() != 1)
	{
		cvtColor(frame, gray, CV_RGB2GRAY);
	}

	Rect searchWindow;
	searchWindow.width = this->location.width * scale;
	searchWindow.height = this->location.height * scale;
	searchWindow.x = this->location.x + this->location.width * 0.5
		- searchWindow.width * 0.5;
	searchWindow.y = this->location.y + this->location.height * 0.5
		- searchWindow.height * 0.5;
	searchWindow &= Rect(0, 0, frame.cols, frame.rows);

	Mat similarity;
	matchTemplate(gray(searchWindow), this->model, similarity, CV_TM_CCOEFF_NORMED);
	double mag_r;
	Point point;
	minMaxLoc(similarity, 0, &mag_r, 0, &point);
	this->location.x = point.x + searchWindow.x;
	this->location.y = point.y + searchWindow.y;

	this->model = gray(location);
	return this->location;
}

Rect Template::getLocation()
{
	return this->location;
}


void limitRect(Rect &location, Size sz)
{
	Rect window(Point(0, 0), sz);
	location = location & window;
}


bool judge_color_rgb(Mat src);
bool judge_color_hsv(Mat src);
vector<Rect> color_filter(Mat frame, vector<Rect> boards, bool color_flag);	//color filter
void save_location(Rect location, int frame_num, Mat frame);
void save_board(vector<Rect> boards, int frame_num, Mat frame);

int main(int argc, char * argv[])
{
	//VideoWriter output_dst("newtracker.avi", CV_FOURCC('M', 'J', 'P', 'G'), 60, Size(640, 480), 1);
	Mat frame;
	bool color_flag = BLUE;
	//color_flag = get_color();


	Template tracker;
	VideoCapture capture;
	capture.open("new.avi");
	//capture.open("red2.mp4");
	if (!capture.isOpened())
	{
		cout << "fail to open" << endl;
		exit(0);
	}
	String cascade_name = "cascade.xml";
	CascadeClassifier detector;
	if (!detector.load(cascade_name))
	{
		printf("--(!)Error loading face cascade\n");
		return -1;
	};


	int64 tic, toc;
	double time = 0;
	bool show_visualization = true;
	int status = 0;     //0：没有目标，1：找到目标进行跟踪

	int frame_num = 0;
	Rect location;
	vector<Rect_<float>> result_rects;
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}
		//resize(frame, frame, Size(), 0.6, 0.8);
		frame_num++;
		tic = getTickCount();
		if (status == 0)
		{
			//cout << "[debug] " << frame_num << ":" << " 没有目标" << endl;
			vector<Rect> boards;
			Mat frame_gray;
			cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
			detector.detectMultiScale(frame_gray, boards, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(10, 10), Size(200, 200));
			boards = color_filter(frame, boards, color_flag);
			save_board(boards, frame_num, frame);
			if (boards.size() > 0)
			{
				cout << "[debug] " << frame_num << ":" << " Detection find " << boards.size() << " objects" << endl;
				if (boards.size() == 1)
					location = boards[0];
				else
				{
					//这个非极大值抑制NMS是基于面积area的
					int max_area = boards[0].width * boards[0].height;
					int max_index = 0;
					for (int i = 1; i < boards.size(); i++)
					{
						int area = boards[i].width * boards[i].height;
						if (area > max_index)
						{
							max_area = area;
							max_index = i;
						}
					}
					location = boards[max_index];
				}
				//save_location(location, frame_num, frame);
				tracker.initTracking(frame, location);
				status = 1;
				cout << "[debug] " << frame_num << ":" << " Start tracking" << endl;
			}
		}
		else if (status == 1)
		{
			location = tracker.track(frame);
			limitRect(location, frame.size());
			if (location.area() == 0)
			{
				status = 0;
				continue;
			}
			// if(frame_num % 3 == 0)
			// {
			//     Mat roi = frame(location);
			//     imwrite("../data/" + to_string(frame_num) + ".jpg", roi);
			// }
			result_rects.push_back(location);
			if (frame_num % 20 == 0)
			{
				//在图片周围进行寻找
				int factor = 2;
				int newx = location.x + (1 - factor) * location.width / 2;
				int newy = location.y + (1 - factor) * location.height / 2;
				Rect loc = Rect(newx, newy, location.width * factor, location.height * factor);
				limitRect(loc, frame.size());
				Mat roi = frame(loc);
				cvtColor(roi, roi, COLOR_BGR2GRAY);
				//imshow("debug", roi);
				vector<Rect> boards;
				detector.detectMultiScale(roi, boards, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(), roi.size());

				if (boards.size() <= 0)
				{
					status = 0;
					cout << "[debug] " << frame_num << ": " << "Tracking loss objects" << endl;
				}
				else
				{
					//location = boards[0];
					location = Rect(boards[0].x + loc.x, boards[0].y + loc.y, boards[0].width, boards[0].height);
					tracker.initTracking(frame, location);
				}
			}
		}
		toc = getTickCount() - tic;
		time += toc;

		if (show_visualization) {
			putText(frame, to_string(frame_num), Point(20, 40), 6, 1,
				Scalar(0, 255, 255), 2);
			if (status == 1)
				rectangle(frame, location, Scalar(0, 128, 255), 2);
			imshow("detectracker", frame);
			//output_dst << frame;

			char key = waitKey(10);
			if (key == 27 || key == 'q' || key == 'Q')
				break;
		}
	}

	time = time / double(getTickFrequency());
	double fps = double(frame_num) / time;
	cout << "fps:" << fps << endl;
	destroyAllWindows();

	return 0;
}



bool judge_color_rgb(Mat src)
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
	if (red_count > blue_count)
		return true;
	else
		return false;
}

//HSV颜色空间颜色判别，注意因为共享内存地址，判别后需要返回BGR空间，否则影响追踪模板。
bool judge_color_hsv(Mat src)
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
	if (red_count > blue_count)
		return true;
	else
		return false;
}

vector<Rect> color_filter(Mat frame, vector<Rect> boards, bool color_flag)	//color filter
{
	vector<Rect> results;
	for (int i = 0; i < boards.size(); i++)
	{
		Mat roi = frame(boards[i]);
		//imshow("roi", roi);
		//waitKey(0);
		bool flag = judge_color_rgb(roi);
		if (flag == color_flag)
			results.push_back(boards[i]);
	}
	//cout << results.size() << endl;
	return results;
}

void save_location(Rect location, int frame_num, Mat frame)
{
	Mat frame1 = frame;
	rectangle(frame1, cvPoint(cvRound(location.x), cvRound(location.y)),
		cvPoint(cvRound((location.x + location.width-1)), cvRound((location.y + location.height-1))),
		Scalar(0,255,255), 3, 8, 0);
	char buf[30];
	sprintf_s(buf, "D:\\frame\\%d.jpg", frame_num);
	imwrite(buf, frame1);
}

void save_board(vector<Rect> boards, int frame_num, Mat frame)
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
