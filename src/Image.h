#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

using namespace std;
using namespace cv;

#ifndef __IMAGE_H__
#define __IMAGE_H__

template<typename aType>
class Image
{
public:
	virtual int GetChannels() = 0;
	virtual int GetHeight() = 0;
	virtual int GetWidth() = 0;
	virtual int GetType() = 0;
	virtual aType Get(int, int, int) = 0;
};

template<typename aType>
class WriteableImage : public Image<aType>
{
public:
	virtual void Set(int, int, int, aType) {}
	virtual void Reset(int, int, char) {};
};

#endif //__IMAGE_H__