#include <vector>
#include <cmath>

using namespace std;

#ifndef __VECTORS_H__
#define __VECTORS_H__

class Vectors
{
private:
	int dimension;
	int height;
	int width;
	int level;
	int size;

	// float* x;
	vector<float> v;

public:
	Vectors():dimension(0), height(0), width(0), level(0) {}
	Vectors(int, int, int, int);
	Vectors(int);
	~Vectors();

	// void Free();
	float Get(int, int, int, int);
	void Set(int, int, int, int, float);
	int Dimension();
};

#endif //__VECTORS_H__