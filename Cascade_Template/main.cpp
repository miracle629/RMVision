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



Mat frame;
Rect2d location;
Rect searchWindow;
Mat search_frame;
Point2d searchWindowCenter;
Point2d VirtualLeftUp;
int flag3;
int flag4;

void limitRect(cv::Rect &location, cv::Size sz)
{
	cv::Rect window(cv::Point(0, 0), sz);
	location = location & window;
}

int main(int argc, char * argv[])
{
	//VideoWriter output_dst("detectracker.avi", CV_FOURCC('M', 'J', 'P', 'G'), 23, Size(640, 480), 1);
	//STAPLE_TRACKER staple;

	// List of tracker types in OpenCV 3.2
	// NOTE : GOTURN implementation is buggy and does not work.
	string trackerTypes[6] = { "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN" };
	// vector <string> trackerTypes(types, std::end(types));

	// Create a tracker
	string trackerType = trackerTypes[2];

	cv::Ptr<cv::Tracker> tracker;

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
	cv::String cascade_name = "cascade.xml";;
	cv::CascadeClassifier detector;
	if (!detector.load(cascade_name))
	{
		printf("--(!)Error loading face cascade\n");
		return -1;
	};


	int64 tic, toc;
	double time = 0;
	bool show_visualization = true;
	int status = 0;     //0：没有目标，1：找到目标进行跟踪
						//Mat frame;
	int frame_num = 0;
	
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}
		frame_num++;
		tic = cv::getTickCount();
		if (status == 0)
		{
			//cout << "[debug] " << frame_num << ":" << " 没有目标" << endl;
			std::vector<Rect> boards;
			cv::Mat frame_gray;
			cv::cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
			detector.detectMultiScale(frame_gray, boards, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30), Size(150, 150));
			if (boards.size() > 0)
			{
				//cout << "[debug]" << frame_num << ":" << "cascade找到" << boards.size() << "个目标" << endl;
				
				cv::imshow("detectracker", frame);
				//waitKey(0);
				if (boards.size() == 1)
					location = boards[0];
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
				}

				rectangle(frame, cvPoint(cvRound(location.x), cvRound(location.y)),
					cvPoint(cvRound((location.x + location.width - 1)), cvRound((location.y + location.height - 1))),
					Scalar(20, 20, 20), 3, 8, 0);
				
				status = 1;
				cout << "wow" << endl;
			}
		}
		else if (status == 1)
		{
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
			cout << "flag3:" << flag3 << ",flag4" << flag4 << endl;
			cout << "::" << searchWindowCenter.x << "::" << searchWindowCenter.y << endl;
			cout << "::" << VirtualLeftUp.x << "::" << VirtualLeftUp.y << endl;
			cout << "::" << searchWindow.x << "::" << searchWindow.y << endl;
			cout << "::" << location.x << "::" << location.y << endl;

			cout << "ok1" << endl;
			search_frame = frame(searchWindow);
			cout << "ok2" << endl;


			cout << "ok3" << endl;
			imshow("search1", search_frame);


			tracker->init(search_frame, location);

			
			bool ok = tracker->update(search_frame, location);
			cout << "::::" << ok << endl;
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
			cout << "::" << location.x << "::" << location.y << endl;
			//limitRect(location, frame.size());
			if (location.empty())
			{
				status = 0;
				continue;
			}
			// if(frame_num % 3 == 0)
			// {
			//     Mat roi = frame(location);
			//     imwrite("../data/" + to_string(frame_num) + ".jpg", roi);
			// }
			cout << "wow12" << endl;
			//staple.tracker_staple_train(frame, false);
			
			if (frame_num % 10 == 0)
			{
				Mat frame_gray;
				cvtColor(frame, frame_gray, COLOR_BGR2BGRA);
				std::vector<Rect> boards;
				detector.detectMultiScale(frame_gray, boards, 1.1, 2,
					0 | CASCADE_SCALE_IMAGE, Size(30, 30), Size(150, 150));
				cout << "[debug] boards: " << boards.size() << endl;
				cout << "gof ball" << endl;
				if (boards.size() <= 0)
				{
					status = 0;
				}
				else if (boards.size() > 0)
				{
					//cout << "[debug]" << frame_num << ":" << "cascade找到" << boards.size() << "个目标" << endl;
					
					
						
					
					cv::imshow("detectracker", frame);
					//waitKey(0);
					if (boards.size() == 1)
						location = boards[0];
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
					}
					rectangle(frame, cvPoint(cvRound(location.x), cvRound(location.y)),
						cvPoint(cvRound((location.x + location.width - 1)), cvRound((location.y + location.height - 1))),
						Scalar(20, 20, 20), 3, 8, 0);
					status = 1;
					cout << "wow" << endl;
				}
			}
		}
		toc = cv::getTickCount() - tic;
		time += toc;

		if (show_visualization) {
			cv::putText(frame, std::to_string(frame_num), cv::Point(20, 40), 6, 1,
				cv::Scalar(0, 255, 255), 2);
			if (status == 1)
				cv::rectangle(frame, location, cv::Scalar(0, 128, 255), 2);
			cv::imshow("detectracker", frame);
			//output_dst << frame;

			char key = cv::waitKey(10);
			if (key == 27 || key == 'q' || key == 'Q')
				break;
		}
	}

	time = time / double(cv::getTickFrequency());
	double fps = double(frame_num) / time;
	std::cout << "fps:" << fps << std::endl;
	cv::destroyAllWindows();

	return 0;
}