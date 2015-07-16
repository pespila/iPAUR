#include <cmath>
#include "../Image/Image.h"
#include "../Parameter/Parameter.h"
#include "Vector3D.h"

#ifndef __PRIMALDUALALGORITHM_H__
#define __PRIMALDUALALGORITHM_H__

class PrimalDualAlgorithm
{
private:
	int steps;
	int height;
	int width;
	int level;
	int size;
	
	float* solution;
	float* f;
	float* u;
	float* u_n;
	float* u_bar;
	float* gradient_transpose;

	Vector3D p;
	Vector3D x;
	Vector3D y;
	Vector3D q;
	Vector3D p_dual;
	Vector3D gradient;
	
	void ScaleArray(float*, float, float*);
	void AddArray(float*, float, float*, float, float*);
	void ScaleVector3D(Vector3D&, float, Vector3D&);
	void AddVector3D(Vector3D&, float, Vector3D&, float, Vector3D&);
	void Nabla(Vector3D&, float*);
	void ProjectionOntoParabola(Vector3D&, Vector3D&);
	void ProjectionOntoConvexCones(Vector3D&, Vector3D&, float, int, int);
	void DykstraAlgorithm(Vector3D&, Vector3D&, float*, float, float, float, int);
	void NablaTranspose(float*, Vector3D&);
	void TruncationOperation(float*, float*);
	void ComputeIsosurface(float*);
	void SetSolution(WriteableImage&);
	void Initialize(Image&);

public:
	PrimalDualAlgorithm():steps(0), height(0), width(0), level(0), size(0), f(NULL), u(NULL), u_n(NULL), u_bar(NULL), gradient_transpose(NULL) {}
	PrimalDualAlgorithm(Image&, int, int);
	~PrimalDualAlgorithm();

	void PrimalDual(Image&, WriteableImage&, Parameter&, int);
};

#endif //__PRIMALDUALALGORITHM_H__