#include <cmath>

#ifndef __VECTOR3D_H__
#define __VECTOR3D_H__

class Vector3D
{
private:
	int dimension;
	int height;
	int width;
	int level;
	int size;

	float* x;

public:
	Vector3D():dimension(3), height(0), width(0), level(0), x(NULL) {}
	Vector3D(int, int, int);
	~Vector3D();

	void Free();
	float Get(int, int, int, int);
	void Set(int, int, int, int, float);
};

#endif //__VECTOR3D_H__