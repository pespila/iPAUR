#include "Image.h"
#include "Vector.h"

#ifndef __LINEAROPERATOR_H__
#define __LINEAROPERATOR_H__

template<class F>
class LinearOperator
{
public:
	LinearOperator() {}
	~LinearOperator() {}

	void Nabla(Vector<F>& dst, Vector<F>& src, int i, int j, int k) {
		if (3 * src.Dimension() == dst.Dimension()) {
			dst.Set(i, j, k, 0, i + 1 < src.Height() ? src.Get(i+1, j, k, 0) - src.Get(i, j, k, 0) : 0.0);
			dst.Set(i, j, k, 1, j + 1 < src.Width() ? src.Get(i, j+1, k, 0) - src.Get(i, j, k, 0) : 0.0);
			dst.Set(i, j, k, 2, k + 1 < src.Level() ? src.Get(i, j, k+1, 0) - src.Get(i, j, k, 0) : 0.0);
		} else {
			cout << "ERROR 03 (Nabla): Dimension of 'dst' is not a third multiple of 'src'!" << endl;
		}
	}
	void NablaTranspose(Vector<F>& dst, Vector<F>& src, int i, int j, int k) {
		if (src.Dimension() == 3 * dst.Dimension()) {
			F value = 0.0;
			value += (i > 0 ? src.Get(i-1, j, k, 0) : 0.0) - (i + 1 < dst.Height() ? src.Get(i, j, k, 0) : 0.0);
			value += (j > 0 ? src.Get(i, j-1, k, 1) : 0.0) - (j + 1 < dst.Width() ? src.Get(i, j, k, 1) : 0.0);
			value += (k > 0 ? src.Get(i, j, k-1, 2) : 0.0) - (k + 1 < dst.Level() ? src.Get(i, j, k, 2) : 0.0);
			dst.Set(i, j, k, 0, value);
		} else {
			cout << "ERROR 04 (NablaTranspose): Dimension of 'src' is not a third multiple of 'dst'!" << endl;
		}
	}
	void Isosurface(Image<F>& dst, Vector<F>& src) {
		if (src.Height() == dst.Height() && src.Width() == dst.Width()) {
			for (int i = 0; i < src.Height(); i++) {
				for (int j = 0; j < src.Width(); j++) {
					for (int k = 0; k < src.Level()-1; k++) {
						F uk0 = src.Get(i, j, k, 0);
						F uk1 = src.Get(i, j, k+1, 0);
						if (uk0 > 0.5 && uk1 <= 0.5) {
							dst.Set(i, j, (0.5 - uk0) / (uk1 - uk0) + (F)(k + 1) / (F)(src.Level()));
							break;
						} else {
							dst.Set(i, j, (F)0);
						}
					}
				}
			}
		} else {
			cout << "ERROR 05 (Isosurface): Height and Width of 'src' and 'dst' do not match!" << endl;
		}
	}
};

#endif //__LINEAROPERATOR_H__