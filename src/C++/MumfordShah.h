#include <cmath>
#include <iostream>
#include "Image.h"

using namespace std;

#ifndef __MUMFORDSHAH_H__
#define __MUMFORDSHAH_H__

template<typename aType>
class MumfordShah
{
private:
	int steps;
	int height;
	int width;
	int channel;
	int size;
	int cSize;
	int pSize;
	int level;
	int proj;
	aType* f;
	aType* u;
	aType* u_n;
	aType* u_bar;
	aType* p_x;
	aType* p_y;
	aType* p_z;
	aType* s_x;
	aType* s_y;
	aType* mu_x;
	aType* mu_y;
	aType* mu_bar_x;
	aType* mu_bar_y;
	aType* mu_n_x;
	aType* mu_n_y;

	void Initialize(Image<aType>&);
	aType Bound(aType, aType, aType, aType, int);
	void Parabola(aType*, aType*, aType*, aType, aType, aType, aType, aType, int, int);
	void ParabolaProjection(aType*, aType*, aType*, aType*, aType*, aType*, aType*, aType, aType);
	void EuclideanProjection(aType*, aType*, aType*, aType*, aType, aType);
	void UpdateMu(aType*, aType*, aType*, aType*, aType*, aType*);
	void Clipping(aType*, aType*, aType*, aType*, aType*, aType);
	void Extrapolation(aType*, aType*, aType*, aType*, aType*, aType*, aType*, aType*, aType*);
	void Isosurface(aType*, aType*);
	void SetSolution(Image<aType>&, aType*);
	
public:
	MumfordShah():steps(0), height(0), width(0), channel(0), size(0), f(NULL), u(NULL), u_n(NULL), u_bar(NULL), p_x(NULL), p_y(NULL), p_z(NULL), s_x(NULL), s_y(NULL), mu_x(NULL), mu_y(NULL), mu_bar_x(NULL), mu_bar_y(NULL), mu_n_x(NULL), mu_n_y(NULL) {}
	MumfordShah(Image<aType>&, int, int);
	~MumfordShah();

	void Minimizer(Image<aType>&, Image<aType>&, aType, aType);
};

#include "MumfordShah.tpp"

#endif //__MUMFORDSHAH_H__