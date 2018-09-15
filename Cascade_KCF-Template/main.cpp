#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

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

class Template {
public:
	Template();
	void initTracking(Mat frame, Rect box, int scale = 2);
	Rect track(Mat frame);
	Rect getLocation();
private:
	Mat model;
	Rect location;
	int scale;
};

Template::Template()
{
	this->scale = 2;
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
//void save_location(Rect location, int frame_num, Mat frame);
//void save_board(vector<Rect> boards, int frame_num, Mat frame);
//vector<int> CalculateXYPixel(Rect location);
bool rectA_intersect_rectB(cv::Rect rectA, cv::Rect rectB);

int main(int argc, char * argv[])
{
	Mat frame;
	Rect2d location;
	Rect searchWindow;
	Mat search_frame;
	Point searchWindowCenter;
	Point VirtualLeftUp;
	int flag3;
	int flag4;


	Rect location1;
	Rect location2;
	//VideoWriter output_dst("detectracker.avi", CV_FOURCC('M', 'J', 'P', 'G'), 23, Size(640, 480), 1);
	//STAPLE_TRACKER staple;

	// List of tracker types in OpenCV 3.3.1
	// NOTE : GOTURN implementation is buggy and does not work.
	string trackerTypes[6] = { "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN" };
	// vector <string> trackerTypes(types, std::end(types));

	// Create a Tracker
	string trackerType = trackerTypes[2];

	Ptr<Tracker> tracker;
	Template tracker1;

	if (trackerType == "BOOSTING")
		tracker = cv::TrackerBoosting::create();
	if (trackerType == "MIL")
		tracker = cv::TrackerMIL::create();
	if (trackerType == "KCF")
		tracker = cv::TrackerKCF::create();
	if (trackerType == "TLD")
		tracker = cv::TrackerTLD::create();
	if (trackerType == "MEDIANFLOW")
		tracker = cv::TrackerMedianFlow::create();
	if (trackerType == "GOTURN")
		tracker = cv::TrackerGOTURN::create();


	cv::VideoCapture capture;
	capture.open("new.avi");
	if (!capture.isOpened())
	{
		std::cout << "fail to open" << std::endl;
		exit(0);
	}
	String cascade_name = "cascade.xml";;
	CascadeClassifier detector;
	if (!detector.load(cascade_name))
	{
		printf("--(!)Error loading face cascade\n");
		return -1;
	};

	int64 tic, toc;
	double time = 0;
	int frame_num = 0;
	bool show_visualization = true;
	int status = 0;     //0：没有目标，1：找到目标进行跟踪
						
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}
		frame_num++;
		tic = getTickCount();
		if (status == 0)
		{
			vector<Rect> boards;
			Mat frame_gray;
			cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
			detector.detectMultiScale(frame_gray, boards, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30), Size(150, 150));
			if (boards.size() > 0)
			{
				if (boards.size() == 1)
				{
					location = boards[0];
					location1 = boards[0];
				}	
				else
				{
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
					location1 = boards[max_index];
				}
				tracker1.initTracking(frame, location1);
				rectangle(frame, cvPoint(cvRound(location.x), cvRound(location.y)),
					cvPoint(cvRound((location.x + location.width - 1)), cvRound((location.y + location.height - 1))),
					Scalar(20, 20, 20), 3, 8, 0);
				status = 1;
			}
		}
		else if (status == 1)
		{
			location1 = tracker1.track(frame);
			limitRect(location1, frame.size());
			searchWindow.width = 150;
			searchWindow.height = 150;
			searchWindowCenter.x = location.x + location.width*0.5;
			searchWindowCenter.y = location.y + location.height*0.5;
			VirtualLeftUp.x = location.x + location.width*0.5 - searchWindow.width*0.5;
			VirtualLeftUp.y = location.y + location.height*0.5 - searchWindow.height*0.5;
			flag3 = 0;
			flag4 = 0;
			if (searchWindowCenter.x >= 75 && searchWindowCenter.x <= 565)
			{
				searchWindow.x = VirtualLeftUp.x;
				location.x -= searchWindow.x;
				flag3 = 1;
			}
			else if (searchWindowCenter.x < 75)
			{
				searchWindow.x = 0;
				flag3 = 2;
			}
			else
			{
				searchWindow.x = 490;
				location.x -= 490;
				flag3 = 3;
			}


			if (searchWindowCenter.y >= 75 && searchWindowCenter.y <= 405)
			{
				searchWindow.y = VirtualLeftUp.y;
				location.y -= searchWindow.y;
				flag4 = 1;
			}
			else if (searchWindowCenter.y < 75)
			{
				searchWindow.y = 0;
				flag4 = 2;
			}
			else
			{
				searchWindow.y = 330;
				location.y -= 330;
				flag4 = 3;
			}
			
			search_frame = frame(searchWindow);
			imshow("search1", search_frame);
			tracker->init(search_frame, location);


			bool ok = tracker->update(search_frame, location);

			if (flag3 == 1)
				location.x += searchWindow.x;
			else if (flag3 == 3)
				location.x += 490;
			else
				location.x += 0;

			if (flag4 == 1)
				location.y += searchWindow.y;
			else if (flag4 == 3)
				location.y += 330;
			else
				location.y += 0;

			if ((location.empty()) || (location1.empty()))
			{
				status = 0;
				continue;
			}
			location2.x = int(location.x);
			location2.y = int(location.y);
			location2.width = int(location.width);
			location2.height = int(location.height);
			bool inc = rectA_intersect_rectB(location2, location1);
			if (inc == false)
			{
				//status = 0;
				//continue;
				
				Mat frame_gray;
				cvtColor(frame, frame_gray, COLOR_BGR2BGRA);
				vector<Rect> boards;
				detector.detectMultiScale(frame_gray, boards, 1.1, 2,
					0 | CASCADE_SCALE_IMAGE, Size(30, 30), Size(150, 150));
				if (boards.size() <= 0)
				{
					status = 0;
				}
				else if (boards.size() > 0)
				{
					if (boards.size() == 1)
					{
						location = boards[0];
						location1 = boards[0];
					}
					else
					{
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
						location1 = boards[max_index];
					}
					tracker1.initTracking(frame, location1);
					rectangle(frame, cvPoint(cvRound(location.x), cvRound(location.y)),
						cvPoint(cvRound((location.x + location.width - 1)), cvRound((location.y + location.height - 1))),
						Scalar(20, 20, 20), 3, 8, 0);
					status = 1;
				}
				
			}
			else if (frame_num % 10 == 0)
			{
				Mat frame_gray;
				cvtColor(frame, frame_gray, COLOR_BGR2BGRA);
					std::vector<Rect> boards;
					detector.detectMultiScale(frame_gray, boards, 1.1, 2,
						0 | CASCADE_SCALE_IMAGE, Size(30, 30), Size(150, 150));
					if (boards.size() <= 0)
					{
						status = 0;
					}
					else if (boards.size() > 0)
					{
						if (boards.size() == 1)
						{
							location = boards[0];
							location1 = boards[0];
						}
						else
						{
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
							location1 = boards[max_index];
						}
						tracker1.initTracking(frame, location1);
						rectangle(frame, cvPoint(cvRound(location.x), cvRound(location.y)),
							cvPoint(cvRound((location.x + location.width - 1)), cvRound((location.y + location.height - 1))),
							Scalar(20, 20, 20), 3, 8, 0);
						status = 1;
					}
				}
			}
			toc = cv::getTickCount() - tic;
			time += toc;

			if (show_visualization) 
			{
				cv::putText(frame, std::to_string(frame_num), cv::Point(20, 40), 6, 1,
					cv::Scalar(0, 255, 255), 2);
				if (status == 1)
					cv::rectangle(frame, location, cv::Scalar(0, 128, 255), 2);
				imshow("detectracker", frame);
				//output_dst << frame;

				char key = cv::waitKey(10);
				if (key == 27 || key == 'q' || key == 'Q')
					break;
			}
		}

		time = time / double(cv::getTickFrequency());
		double fps = double(frame_num) / time;
		cout << "fps:" << fps << std::endl;
		cv::destroyAllWindows();

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
	//cout << "[debug] " << "blue_count: " << blue_count << "\tred_count: " << red_count << endl;
	if ((red_count - blue_count) >= 50)
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
	//cout << "[debug] " << "blue_count: " << blue_count << "\tred_count: " << red_count << endl;
	if ((red_count - blue_count) >= 50)
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
		
		bool flag = judge_color_rgb(roi);
		if (flag == color_flag)
			results.push_back(boards[i]);
	}
	return results;
}


bool rectA_intersect_rectB(cv::Rect rectA, cv::Rect rectB)
{
	if (rectA.x > rectB.x + rectB.width) { return false; }
	if (rectA.y > rectB.y + rectB.height) { return false; }
	if ((rectA.x + rectA.width) < rectB.x) { return false; }
	if ((rectA.y + rectA.height) < rectB.y) { return false; }

	return true;
}
