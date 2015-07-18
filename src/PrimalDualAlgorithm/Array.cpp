#include "Vector3D.h"

Vector3D::Vector3D(int height, int width, int level) {
	this->height = height;
	this->width = width;
	this->level = level;
	this->dimension = 3;
	this->size = dimension * height * width * level;
	this->x = new float[size];
}

Vector3D::~Vector3D() {
	this->x = NULL;
}

void Vector3D::Free() {
	delete[] this->x;
}

float Vector3D::Get(int i, int j, int k, int c) {
	return this->x[j + i * width + k * height * width + c * height * width * dimension];
}

void Vector3D::Set(int i, int j, int k, int c, float value) {
	this->x[j + i * width + k * height * width + c * height * width * dimension] = value;
}