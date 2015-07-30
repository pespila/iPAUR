#include <iostream>
#include <vector>

using namespace std;

#ifndef __SUBIMAGE_H__
#define __SUBIMAGE_H__

template<class F>
class SubImage
{
private:
	vector<F> v;
	int height;
	int width;
	int level;
public:
	SubImage():height(0), width(0) {}
	SubImage(int height, int width, int level, F value = (F)0) {
		v.resize(height * width, value);
		this->height = height;
		this->width = width;
		this->level = level;
	}
	~SubImage() {v.clear();}

	F Get(int i, int j, int k) {return this->v[j + i * width + k * height * width];}
	void Set(int i, int j, int k, F value) {this->v[j + i * width + k * height * width] = value;}
	
	int Height() {return this->height;}
	int Width() {return this->width;}
	int Level() {return this->level;}
	int Size() {return this->v.size();}
	
	void Print() {
		for (int k = 0; k < level; k++) {
			cout << "Level: " << k << endl;
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					cout << Get(i, j) << " ";
				}
				cout << endl;
			}
			cout << endl;
		}
	}
};

#endif //__SUBIMAGE_H__