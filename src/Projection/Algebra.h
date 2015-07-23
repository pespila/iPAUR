#include "Vector.h"
#include <cmath>

#ifndef __ALGEBRA_H__
#define __ALGEBRA_H__

template<class F>
class Algebra
{
public:
	Algebra() {}
	~Algebra() {}

	void AddVector(Vector<F>& dst, Vector<F>& src1, Vector<F>& src2, F factor1, F factor2) {
		if (!(dst.EqualProperties(src1) && dst.EqualProperties(src2))) {
			cout << "ERROR 01 (AddVector): Height, Width, Level and/or Dimension do not match for used vectors!" << endl;
		} else {
			for (int c = 0; c < src1.Dimension(); c++)
				for (int k = 0; k < src1.Level(); k++)
					for (int i = 0; i < src1.Height(); i++)
						for (int j = 0; j < src1.Width(); j++)
							dst.Set(i, j, k, c, factor1 * src1.Get(i, j, k, c) + factor2 * src2.Get(i, j, k, c));
		}
	}

	void ScaleVector(Vector<F>& dst, Vector<F>& src, F factor) {
		if (!(dst.EqualProperties(src))) {
			cout << "ERROR 02 (ScaleVector): Height, Width, Level and/or Dimension do not match for used vectors!" << endl;
		} else {
			for (int c = 0; c < src.Dimension(); c++)
				for (int k = 0; k < src.Level(); k++)
					for (int i = 0; i < src.Height(); i++)
						for (int j = 0; j < src.Width(); j++)
							dst.Set(i, j, k, c, factor * src.Get(i, j, k, c));
		}
	}
};

#endif //__ALGEBRA_H__