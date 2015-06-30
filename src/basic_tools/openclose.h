#include "image.h"
#include "dilatation.h"
#include "erosion.h"

#ifndef __OPENCLOSE_H__
#define __OPENCLOSE_H__

void open(Image&, WriteableImage&, int*, int);
void close(Image&, WriteableImage&, int*, int);

#endif //__OPENCLOSE_H__