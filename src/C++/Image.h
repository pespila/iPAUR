#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

using namespace std;
using namespace cv;

#ifndef __IMAGE_H__
#define __IMAGE_H__

template<typename aType>
class Image
{
public:
	Image():image(NULL), height(0), width(0), channels(0), type(0) {}
	Image(const string, bool);
	~Image();
	int Channels() {return this->channels;}
	int Height() {return this->height;}
    int Width() {return this->width;}
	int Type() {return this->type;}
	aType Get(int i, int j, int k) {return this->image[j + i * this->width + k * this->height * this->width];}
	void Set(int i, int j, int k, aType value) {this->image[j + i * this->width + k * this->height * this->width] = value;}

	void Read(const string, bool);
	void Write(const string);
	void Reset(int, int, int, char);

protected:
	aType* image;
	int height;
	int width;
	int channels;
	char type;
};

#include "Image.tpp"

#endif //__IMAGE_H__