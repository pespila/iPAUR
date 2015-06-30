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
	virtual unsigned short get_channels() = 0;
	virtual unsigned short get_height() = 0;
	virtual unsigned short get_width() = 0;
	virtual unsigned short get_type() = 0;
	virtual unsigned char get_pixel(int, int, int) = 0;
};

class WriteableImage : public Image
{
public:
	virtual void set_pixel(int, int, int, unsigned char) {}
	virtual void reset_image(unsigned short, unsigned short, char) {};
};

#endif //__IMAGE_H__