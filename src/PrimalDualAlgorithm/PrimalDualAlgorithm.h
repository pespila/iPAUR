#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include "../Image/Image.h"
#include "../Parameter/Parameter.h"
#include "../Util/Util.h"

#ifndef __PRIMALDUALALGORITHM_H__
#define __PRIMALDUALALGORITHM_H__

struct Vector3D
{
	float* x1;
	float* x2;
	float* x3;
};

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
	// struct Vector3D gradient;
	struct Vector3D p;
	struct Vector3D x;
	struct Vector3D y;
	struct Vector3D q;
	struct Vector3D gradient;
	struct Vector3D p_dual;
	float* gradient_transpose;
	
	void ScaleArray(float*, float, float*);
	void AddArray(float*, float, float*, float, float*);
	void ScaleVector3D(struct Vector3D, float, struct Vector3D);
	void AddVector3D(struct Vector3D, float, struct Vector3D, float, struct Vector3D);
	void Nabla(struct Vector3D, float*);
	void ProjectionOntoParabola(struct Vector3D, struct Vector3D);
	void ProjectionOntoConvexCone(struct Vector3D, struct Vector3D, float);
	void DykstraAlgorithm(struct Vector3D, struct Vector3D, float*, float, float, float, int);
	void NablaTranspose(float*, struct Vector3D);
	void ComputeIsosurface(float*);
	void SetSolution(WriteableImage&);
	
	void Initialize(Image&);
	float Newton(float, float, float);
	void Nabla(float*, float*, float*, float*);
	void VectorOfInnerProduct(float*, float*, float*, float*);
	void SoftShrinkageScheme(float*, float*, float*, float);
	void ProjectionOntoParabola(float*, float*, float*);
	void ProjectionOntoConvexCone(float*, float*, float*, float);
	void NewtonProjection(float*, float*, float*, float*, float, float);
	void TruncationOperation(float*, float*);
	float Constraint(float, float, float, float, float, int);

public:
	PrimalDualAlgorithm():steps(0), height(0), width(0), level(0), size(0), f(NULL), u(NULL), u_n(NULL), u_bar(NULL) {}
	PrimalDualAlgorithm(Image&, int, int);
	~PrimalDualAlgorithm();

	void PrimalDual(Image&, WriteableImage&, Parameter&, int);
};

#endif //__PRIMALDUALALGORITHM_H__