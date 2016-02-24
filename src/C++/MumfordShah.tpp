template<typename aType>
MumfordShah<aType>::MumfordShah(Image<aType>& src, int steps, int level) {
	this->steps = steps;
	this->channel = src.Channels();
	this->height = src.Height();
	this->width = src.Width();
	this->level = level;
	this->size = height * width * channel;
	this->cSize = height * width * level * channel;
	this->proj = level * (level - 1) / 2 + level;
	this->pSize = height * width * proj * channel;
	this->f = (aType*)malloc(size*sizeof(aType));
	this->u = (aType*)malloc(cSize*sizeof(aType));
	this->u_n = (aType*)malloc(cSize*sizeof(aType));
	this->u_bar = (aType*)malloc(cSize*sizeof(aType));
	this->p_x = (aType*)malloc(cSize*sizeof(aType));
	this->p_y = (aType*)malloc(cSize*sizeof(aType));
	this->p_z = (aType*)malloc(cSize*sizeof(aType));
	this->s_x = (aType*)malloc(pSize*sizeof(aType));
	this->s_y = (aType*)malloc(pSize*sizeof(aType));
	this->mu_x = (aType*)malloc(pSize*sizeof(aType));
	this->mu_y = (aType*)malloc(pSize*sizeof(aType));
	this->mu_bar_x = (aType*)malloc(pSize*sizeof(aType));
	this->mu_bar_y = (aType*)malloc(pSize*sizeof(aType));
	this->mu_n_x = (aType*)malloc(pSize*sizeof(aType));
	this->mu_n_y = (aType*)malloc(pSize*sizeof(aType));
}

template<typename aType>
MumfordShah<aType>::~MumfordShah() {
	free(f);
	free(u);
	free(u_n);
	free(u_bar);
	free(p_x);
	free(p_y);
	free(p_z);
	free(s_x);
	free(s_y);
	free(mu_x);
	free(mu_y);
	free(mu_bar_x);
	free(mu_bar_y);
	free(mu_n_x);
	free(mu_n_y);
}

template<typename aType>
void MumfordShah<aType>::Initialize(Image<aType>& src) {
	aType img_val = 0.f;
	int I, P;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				img_val = (aType)src.Get(i, j, k) / 255.f;
				f[j + i * width + k * height * width] = img_val;
				for (int p = 0; p < proj; p++) {
					I = j + i * width + p * height * width + k * height * width * level;
					P = j + i * width + p * height * width + k * height * width * proj;
					if (p < level) {
						u[I] = img_val;
						u_n[I] = img_val;
						u_bar[I] = img_val;
						p_x[I] = 0.f;
						p_y[I] = 0.f;
						p_z[I] = 0.f;
					}
					mu_x[P] = 0.f;
					mu_y[P] = 0.f;
					mu_bar_x[P] = 0.f;
					mu_bar_y[P] = 0.f;
					mu_n_x[P] = 0.f;
					mu_n_y[P] = 0.f;
					s_x[P] = 0.f;
					s_y[P] = 0.f;
				}
			}
		}
	}
}

template<typename aType>
aType MumfordShah<aType>::Bound(aType x, aType y, aType lambda, aType img, int k) {
	return (0.25f * (x*x+y*y) - lambda * pow(k/level - img, 2));
}

template<typename aType>
void MumfordShah<aType>::Parabola(aType* p_x, aType* p_y, aType* p_z, aType x, aType y, aType z, aType img, aType lambda, int k, int I) {
	aType B = z + lambda * pow(k / level - img, 2);
    aType norm = sqrtf(x*x+y*y);
    aType v = 0.f;
    aType a = 2.f * 0.25f * norm;
    aType b = 2.f / 3.f * (1.f - 2.f * 0.25f * B);
    aType d = b < 0 ? (a - pow(sqrtf(-b), 3)) * (a + pow(sqrtf(-b), 3)) : a*a+b*b*b;
    aType c = pow((a + sqrtf(d)), 1.f/3.f);
    if (d >= 0) {
        v = c == 0 ? 0.f : c - b / c;
    } else {
        v = 2.f * sqrtf(-b) * cos((1.f / 3.f) * acos(a / (pow(sqrtf(-b), 3))));
    }
    p_x[I] = norm == 0 ? 0.f : (v / (2.0 * 0.25f)) * x / norm;
    p_y[I] = norm == 0 ? 0.f : (v / (2.0 * 0.25f)) * y / norm;
    p_z[I] = Bound(p_x[I], p_y[I], lambda, img, k);
}

template<typename aType>
void MumfordShah<aType>::ParabolaProjection(aType* p_x, aType* p_y, aType* p_z, aType* mu_x, aType* mu_y, aType* u_bar, aType* f, aType sigma, aType lambda) {
	aType x, y, z, lx, ly, img, bound, tmp;
	int I, J, K;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				J = j + i * width + k * height * width;
				img = f[J];
				for (int l = 0; l < level; l++) {
					I = j + i * width + l * height * width + k * height * width * level;
					tmp = u_bar[I];
					x = (j+1<width) ? (u_bar[(j+1) + i * width + l * height * width + k * height * width * level] - tmp) : 0.f;
					y = (i+1<height) ? (u_bar[j + (i+1) * width + l * height * width + k * height * width * level] - tmp) : 0.f;
					z = (l+1<level) ? (u_bar[j + i * width + (l+1) * height * width + k * height * width * level] - tmp) : 0.f;
					K = 0;
					lx = 0.f; ly = 0.f;
					for (int k1 = 0; k1 < level; k1++) {
						for (int k2 = k1; k2 < level; k2++) {
							if (l <= k2 && l >= k1) {
								lx += mu_x[j + i * width + K * height * width + k * height * width * level];
								ly += mu_y[j + i * width + K * height * width + k * height * width * level];
							}
							K++;
						}
					}
					x = p_x[I] + sigma * (x - lx);
					y = p_y[I] + sigma * (y - ly);
					z = p_z[I] + sigma * z;
					bound = Bound(x, y, lambda, img, (l+1));
					if (z < bound) {
						Parabola(p_x, p_y, p_z, x, y, z, img, lambda, (l+1), I);
					} else {
						p_x[I] = x;
						p_y[I] = y;
						p_z[I] = z;
					}
				}
			}
		}
	}
}

template<typename aType>
void MumfordShah<aType>::EuclideanProjection(aType* s_x, aType* s_y, aType* mu_bar_x, aType* mu_bar_y, aType sigma, aType nu) {
	aType x, y, norm;
	int I, K;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				K = 0;
				for (int k1 = 0; k1 < level; k1++) {
					for (int k2 = k1; k2 < level; k2++) {
						I = j + i * width + K * height * width + k * height * width * proj;
						x = s_x[I] + sigma * mu_bar_x[I];
						y = s_y[I] + sigma * mu_bar_y[I];
						norm = sqrtf(x*x+y*y);
						s_x[I] = (norm <= nu) ? x : nu*x/norm;
						s_y[I] = (norm <= nu) ? y : nu*y/norm;
						K++;
					}
				}
			}
		}
	}
}

template<typename aType>
void MumfordShah<aType>::UpdateMu(aType* mu_x, aType* mu_y, aType* mu_n_x, aType* mu_n_y, aType* p_x, aType* p_y) {
	aType tau, x, y;
	int I, J, K;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				K = 0;
				for (int k1 = 0; k1 < level; k1++) {
					for (int k2 = k1; k2 < level; k2++) {
						I = j + i * width + K * width * height + k * height * width * level;
						// tau = 1.f/(2.f+(aType)(k2-k1));
						tau = 1.f/(2.f+150.f);
						x = 0.f; y = 0.f;
						for (int l = k1; l <= k2; l++) {
							J = j + i * width + l * width * height + k * width * height * level;
							x += p_x[J];
							y += p_y[J];
						}
						mu_x[I] = mu_n_x[I] - tau * (s_x[I] - x);
						mu_y[I] = mu_n_y[I] - tau * (s_y[I] - y);
						K++;
					}
				}
			}
		}
	}
}

template<typename aType>
void MumfordShah<aType>::Clipping(aType* u, aType* u_n, aType* p_x, aType* p_y, aType* p_z, aType tau) {
	aType x, y, z, tmp;
	int I;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				for (int l = 0; l < level; l++) {
					I = j + i * width + l * width * height + k * width * height * level;
					x = p_x[I] - (j>0 ? p_x[(j-1) + i * width + l * width * height + k * width * height * level] : 0.f);
					y = p_y[I] - (i>0 ? p_y[j + (i-1) * width + l * width * height + k * width * height * level] : 0.f);
					z = p_z[I] - (l>0 ? p_z[j + i * width + (l-1) * width * height + k * width * height * level] : 0.f);
					tmp = u_n[I] + tau * (x+y+z);
					if (l == 0) {
						u[I] = 1.f;
					} else if (l == level-1) {
						u[I] = 0.f;
					} else {
						u[I] = fmin(1.f, fmax(0.f, tmp));
					}
				}
			}
		}
	}
}

template<typename aType>
void MumfordShah<aType>::Extrapolation(aType* u_bar, aType* mu_bar_x, aType* mu_bar_y, aType* u, aType* mu_x, aType* mu_y, aType* u_n, aType* mu_n_x, aType* mu_n_y) {
	int I = 0;
	int P = 0;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				for (int p = 0; p < proj; p++) {
					I = j + i * width + p * height * width + k * height * width * level;
					P = j + i * width + p * height * width + k * height * width * proj;
					if (p < level) {
						u_bar[I] = 2.f * u[I] - u_n[I];
						u_n[I] = u[I];
					}
					mu_bar_x[P] = 2.f * mu_x[P] - mu_n_x[P];
					mu_bar_y[P] = 2.f * mu_y[P] - mu_n_y[P];
					mu_n_x[P] = mu_x[P];
					mu_n_y[P] = mu_y[P];
				}
			}
		}
	}
}

template<typename aType>
void MumfordShah<aType>::Isosurface(aType* f, aType* u) {
	aType val, u0, u1;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				for (int l = 0; l < level-1; l++) {
					u0 = u[j + i * width + l * width * height + k * width * height * level];
					u1 = u[j + i * width + (l+1) * width * height + k * width * height * level];
					if (u0 > 0.5 && u1 <= 0.5) {
						val = ((l+1) + (0.5 - u0) / (u1 - u0)) / level;
						break;
					} else {
						val = u1;
					}
				}
				f[j + i * width + k * width * height] = val;
			}
		}
	}
}

template<typename aType>
void MumfordShah<aType>::SetSolution(Image<aType>& dst, aType* f) {
	for (int k = 0; k < channel; k++)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				dst.Set(i, j, k, f[j + i * width + k * height * width] * 255.f);
}

template<typename aType>
void MumfordShah<aType>::Minimizer(Image<aType>& src, Image<aType>& dst, aType lambda, aType nu) {
	int k;
	aType sigmaP = 1.f/(3.f + level);
	aType sigmaS = 1.f;
	aType tauU = 1.f/6.f;
	dst.Reset(height, width, channel, src.Type());
	Initialize(src);
	for (k = 1; k < steps; k++) {
		ParabolaProjection(p_x, p_y, p_z, mu_x, mu_y, u_bar, f, sigmaP, lambda);
		EuclideanProjection(s_x, s_y, mu_bar_x, mu_bar_y, sigmaS, nu);
		UpdateMu(mu_x, mu_y, mu_n_x, mu_n_y, p_x, p_y);
		Clipping(u, u_n, p_x, p_y, p_z, tauU);
		Extrapolation(u_bar, mu_bar_x, mu_bar_y, u, mu_x, mu_y, u_n, mu_n_x, mu_n_y);
	}
	Isosurface(f, u);
	cout << "Iterations: " << k << endl;
	SetSolution(dst, f);
}