#include <math.h>
#include "Image.h"
#include "Parameter.h"

#ifndef __IPAUR_H__
#define __IPAUR_H__

template<typename aType>
class iPAUR
{
private:
	int steps;
	int height;
	int width;
	int channel;
	int size;
	aType* g;
	aType* u;
	aType* u_n;
	aType* u_bar;
	aType* v;
	aType* v_n;
	aType* v_bar;
	aType* gradient_ux;
	aType* gradient_uy;
	aType* gradient_vx;
	aType* gradient_vy;
	aType* gradient_transpose_u;
	aType* gradient_transpose_v;
	aType* p_x;
	aType* p_y;
	aType* q_x;
	aType* q_y;
	
	void Initialize(Image<aType>&);
	void SetSolution(WriteableImage<aType>&);
	void Nabla(aType*, aType*, aType*, aType*, aType*, aType);
	void ProxDv(aType*, aType*, aType*, aType, aType);
	void ProxRstarV(aType*, aType*, aType*, aType*);
	void NablaTranspose(aType*, aType*, aType*, aType*, aType);
	void ProxDu(aType*, aType*, aType*, aType*, aType, aType, aType);
	void ProxRstarU(aType*, aType*, aType*, aType*);
	void Extrapolation(aType*, aType*, aType*, aType);

public:
	iPAUR():steps(0), height(0), width(0), channel(0), size(0), g(NULL), u(NULL), u_n(NULL), u_bar(NULL), v(NULL), v_n(NULL), v_bar(NULL), gradient_ux(NULL), gradient_uy(NULL), gradient_vx(NULL), gradient_vy(NULL), gradient_transpose_u(NULL), gradient_transpose_v(NULL), p_x(NULL), p_y(NULL), q_x(NULL), q_y(NULL) {}
	iPAUR(Image<aType>&, int);
	~iPAUR();

	void iPAURmodel(Image<aType>&, WriteableImage<aType>&, aType, aType);
};

#include "iPaur.tpp"

#endif //__IPAUR_H__