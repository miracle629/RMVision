#include "Template.h"

/*
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
*/


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

void Template::limitRect(Rect &location, Size sz)
{
	Rect window(Point(0, 0), sz);
	location = location & window;
}

