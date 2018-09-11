#include "NumericProcess.h"



vector<int> CalculateXYPixel(Rect location)
{
	vector<int> XY_Pixel;
	int x_pixel = location.x + location.width - window_width / 2;
	int y_pixel = location.y + location.height - window_height / 2;

	XY_Pixel.push_back(x_pixel);
	XY_Pixel.push_back(y_pixel);
	return XY_Pixel;
}
