#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

#ifndef __IMAGE_H__
#define __IMAGE_H__

class Image
{
public:
	virtual unsigned short get_height() = 0;
	virtual unsigned short get_width() = 0;
	virtual unsigned short get_type() = 0;
	virtual unsigned char get_gray_pixel_at_position(int, int) {return 0;}
	virtual unsigned char get_color_pixel_at_position(int, int, int) {return 0;}
};

class WriteableImage : public Image
{
public:
	virtual void set_gray_pixel_at_position(int, int, unsigned char) {}
	virtual void set_color_pixel_at_position(int, int, int, unsigned char) {}
};

#endif //__IMAGE_H__