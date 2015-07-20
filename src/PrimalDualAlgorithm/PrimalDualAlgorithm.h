#include <cmath>
#include "../Image/Image.h"
#include "../Parameter/Parameter.h"
#include "Vectors.h"

#ifndef __PRIMALDUALALGORITHM_H__
#define __PRIMALDUALALGORITHM_H__

class PrimalDualAlgorithm
{
private:
	int height;
	int width;
	int level;

	Vectors u;
	Vectors u_n;
	Vectors u_bar;
	Vectors gradient_transpose;
	
	Vectors p;
	Vectors x;
	Vectors y;
	Vectors q;
	Vectors p_dual;
	Vectors gradient;
	
	void ScaleVectors(Vectors&, float, Vectors&);
	void AddVectors(Vectors&, float, Vectors&, float, Vectors&);
	void Add(Vectors&, float, Vectors&, float, Vectors&, int, int, int);
	void Nabla(Vectors&, Vectors&);
	void NablaTranspose(Vectors&, Vectors&);
	void TruncationOperation(Vectors&, Vectors&);
	float L2Norm(Vectors&, int);
	void L2Projection(Vectors&, Vectors&, float);
	void SoftShrinkageScheme(Vectors&, Vectors&, int, int, int, int, float);
	void ProjectionOntoParabola(Vectors&, Vectors&, Image&, float, float, int, int, int);
	// void DykstraAlgorithm(Vectors&, Vectors&, Image&, float, float, float, int, int, int, int);
	void DykstraAlgorithm(Vectors&, Vectors&, Image&, float, float, float, int);
	void ComputeIsosurface(WriteableImage&, Vectors&);

public:
	PrimalDualAlgorithm():height(0), width(0), level(0) {}
	PrimalDualAlgorithm(Image&, int);
	~PrimalDualAlgorithm() {}

	void PrimalDual(Image&, WriteableImage&, Parameter&, int, int);
};

#endif //__PRIMALDUALALGORITHM_H__