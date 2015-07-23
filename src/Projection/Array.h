#include <iostream>
#include <vector>

using namespace std;

#ifndef __ARRAY_H__
#define __ARRAY_H__

template<class F>
class Array
{
private:
	vector<F> v;

public:
	Array();
	Array(int size, F value = (F)0) {this->v.resize(size, value);}
	~Array() {this->v.clear();}

	int Dimension() {return this->v.size();}
	F Get(int i) {return this->v[i];}
	void Set(int i, F value) {this->v[i] = value;}
	void Print() {
		for (int i = 0; i < v.size(); i++)
			cout << this->v[i] << " ";
		cout << endl;
	}
};

#endif //__ARRAY_H__