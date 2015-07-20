#include "Vectors.h"

Vectors::Vectors(int height, int width, int level, int dimension) {
	this->height = height;
	this->width = width;
	this->level = level;
	this->dimension = dimension;
	this->size = dimension * height * width * level;
	this->v.resize(this->size, 0.0);
	// this->x = new float[size];
}

Vectors::Vectors(int dimension) {
	this->height = 0;
	this->width = 0;
	this->level = 0;
	this->dimension = dimension;
	this->size = dimension;
	this->v.resize(this->size, 0.0);
}

Vectors::~Vectors() {
	v.clear();
}

// void Vectors::Free() {
// 	delete[] this->x;
// }

float Vectors::Get(int i, int j, int k, int c) {
	if (this->height == 0 && this->width == 0 && this->level == 0) {
		return this->v[c];
	} else {
		return this->v[j + i * width + k * height * width + c * height * width * dimension];
	}
}

void Vectors::Set(int i, int j, int k, int c, float value) {
	if (this->height == 0 && this->width == 0 && this->level == 0) {
		this->v[c] = value;
	} else {
		this->v[j + i * width + k * height * width + c * height * width * dimension] = value;
	}
}

int Vectors::Dimension() {
	return this->dimension;
}