#include "image.h"
#include "util.h"

#ifndef __PRIMAL_DUAL_H__
#define __PRIMAL_DUAL_H__

void dykstra_projection(int, int, int, int, float*);
// void dykstra_projection(float*, float*, float*, float*, int, int);
void truncation_operation(float*, float*, int);
void primal_dual(gray_img*, param*, const char*, int);

#endif //__PRIMAL_DUAL_H__