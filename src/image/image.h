#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

using namespace std;
using namespace cv;

#ifndef __IMAGE_H__
#define __IMAGE_H__

class Image
{
public:
	virtual unsigned short GetChannels() = 0;
	virtual unsigned short GetHeight() = 0;
	virtual unsigned short GetWidth() = 0;
	virtual unsigned short GetType() = 0;
	virtual unsigned char Get(int, int, int) = 0;
};

class WriteableImage : public Image
{
public:
	virtual void Set(int, int, int, unsigned char) {}
	virtual void Reset(unsigned short, unsigned short, char) {};
};

#endif //__IMAGE_H__