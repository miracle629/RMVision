#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>



int main()
{
	cv::cuda::GpuMat image_gpu, image_gpu_gray;
	cv::VideoCapture capture("new.avi");
	cv::Ptr<cv::cuda::CascadeClassifier> cascade_gpu = cv::cuda::CascadeClassifier::create("/home/nvidia/opencv-3.3.1/data/haarcascades_cuda/haarcascade_frontalface_alt_tree.xml");
	cv::Mat frame;
	while(capture.isOpened())
	{
		capture>>frame;
		std::cout<<"ok"<<std::endl;
		if(frame.empty())
			break;
		std::vector<cv::Rect> objects;
		image_gpu.upload(frame);
		cv::cuda::cvtColor(image_gpu, image_gpu_gray, CV_BGR2GRAY);
		cv::cuda::GpuMat objbuf;
		cascade_gpu->detectMultiScale(image_gpu_gray, objbuf);
		cascade_gpu->convert(objbuf, objects);
		
		for(std::vector<cv::Rect>::const_iterator it = objects.begin(); it != objects.end(); ++it) 
			cv::rectangle(frame, *it, cv::Scalar(0,0,255));
	}
	cv::waitKey(0);
	return 0;
}
