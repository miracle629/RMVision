#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Template.h"
#include "ImageProcessing.h"
#include "NumericProcess.h"


#define RED true
#define BLUE false


int main(int argc, char * argv[])
{
	//VideoWriter output_dst("newtracker.avi", CV_FOURCC('M', 'J', 'P', 'G'), 60, Size(640, 480), 1);
	
	bool color_flag = BLUE;
	//bool color_flag = RED;


	VideoCapture capture;
	capture.open("new.avi");//������Ƶ
	//capture.open("red2.mp4");
	//capture.open(0);//��0��������ͷ
	if (!capture.isOpened())
	{
		cout << "fail to open" << endl;
		exit(0);
	}

	
	//����׷������
	Template tracker;

	//����������
	String cascade_name = "cascade.xml";
	CascadeClassifier detector;
	if (!detector.load(cascade_name))
	{
		printf("--(!)Error loading face cascade\n");
		return -1;
	};


	Mat frame;//�����֡
	int frame_num = 0;//֡��������
	Rect location;//ʶ�𵽵�Ŀ�꣬���ο���
	vector<int> XY_pixel;//�����XYƫ��ֵ

	//��ʱ��
	int64 tic, toc;
	double time = 0;
	
	bool show_visualization = true;//��ʾ׷��Ч��
	bool transmit_message = true;//�����ַ�����STM32F4

	int status = 0;     //0��û��Ŀ�꣬1���ҵ�Ŀ����и���

	//����ѭ����ʼ
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}

		frame_num++;
		tic = getTickCount();

		if (status == 0)//����׷�ٵ�һ֡���Ŀ��
		{
			vector<Rect> boards;//�洢���Ŀ��
			Mat frame_gray;
			cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
			detector.detectMultiScale(frame_gray, boards, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30), Size(180, 180));
			boards = ColorFilter(frame, boards, color_flag);//��ɫ�˲��жϵ���
			//save_board(boards, frame_num, frame);
			if (boards.size() > 0)
			{
				if (boards.size() == 1)
					location = boards[0];
				else
				{
					//��������ĵķǼ���ֵ���ƣ�NMS�����ҵ�����Ŀ��
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
				tracker.initTracking(frame, location);//��ʼ��׷������
				status = 1;
				cout << "[debug] " << frame_num << ":" << " Start tracking" << endl;
			}
		}
		else if (status == 1)//�������ٳɹ�����
		{
			location = tracker.track(frame);
			tracker.limitRect(location, frame.size());
			if (location.area() == 0)//׷��ʧ��
			{
				status = 0;
				continue;
			}
			//����25֡׷�ٳɹ����������¼��
			if (frame_num % 25 == 0)
			{
				//��ͼƬ��Χ����Ѱ��
				int factor = 2;
				int newx = location.x + (1 - factor) * location.width / 2;
				int newy = location.y + (1 - factor) * location.height / 2;
				Rect loc = Rect(newx, newy, location.width * factor, location.height * factor);
				tracker.limitRect(loc, frame.size());
				Mat roi = frame(loc);
				cvtColor(roi, roi, COLOR_BGR2GRAY);
				vector<Rect> boards;
				detector.detectMultiScale(roi, boards, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(), roi.size());

				if (boards.size() <= 0)
				{
					status = 0;
					cout << "[debug] " << frame_num << ": " << "Tracking loss objects" << endl;
				}
				else
				{
					boards = ColorFilter(frame, boards, color_flag);//��ɫ�˲��жϵ���
					location = Rect(boards[0].x + loc.x, boards[0].y + loc.y, boards[0].width, boards[0].height);
					tracker.initTracking(frame, location);
					tracker.initTracking(frame, location);
				}
			}
		}
		//��ʱ
		toc = getTickCount() - tic;
		time += toc;

		if (show_visualization) //���ӻ�
		{
			putText(frame, to_string(frame_num), Point(20, 40), 6, 1,
				Scalar(0, 255, 255), 2);
			if (status == 1)
				rectangle(frame, location, Scalar(0, 128, 255), 2);
			imshow("detectracker", frame);

			char key = waitKey(10);
			if (key == 27 || key == 'q' || key == 'Q')
				break;
		}
		/*
		if (transmit_message)
		{
			CSerial sel;
			XY_pixel = CalculateXYPixel(location);
			int X_bias = XY_pixel[0]+320;
			int Y_bias = XY_pixel[1]+240;
			vector<unsigned char> sendData;
			sendData = sel.DataProcess(X_bias, X_ctrl);
			string dev = "/dev/ttyUSB0";
			
			sel.OpenSerialPort(_T("COM6"), 115200, 8, 1);
			string buff = to_string(sendData[0]) + to_string(sendData[1]);
			sel.SendData(buff.data(), buff.length());
			sendData = sel.DataProcess(Y_bias, Y_ctrl);
			buff = to_string(sendData[0]) + " " + to_string(sendData[1]);
			sel.SendData(buff.data(), buff.length());
		}
		*/
		
	}

	//����֡��
	time = time / double(getTickFrequency());
	double fps = double(frame_num) / time;
	cout << "fps:" << fps << endl;
	destroyAllWindows();

	return 0;
}