template<typename aType>
RealTimeMinimizer<aType>::RealTimeMinimizer(Image<aType>& src, int steps) {
	this->steps = steps;
	this->channel = src.Channels();
	this->height = src.Height();
	this->width = src.Width();
	this->size = height * width * channel;
	this->f = (aType*)malloc(size*sizeof(aType));
	this->u = (aType*)malloc(size*sizeof(aType));
	this->u_bar = (aType*)malloc(size*sizeof(aType));
	this->p_x = (aType*)malloc(size*sizeof(aType));
	this->p_y = (aType*)malloc(size*sizeof(aType));
}

template<typename aType>
RealTimeMinimizer<aType>::~RealTimeMinimizer() {
	free(f);
	free(u);
	free(u_bar);
	free(p_x);
	free(p_y);
}

template<typename aType>
void RealTimeMinimizer<aType>::Initialize(Image<aType>& src) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				f[j + i * width + k * height * width] = (aType)src.Get(i, j, k)/255.f;
				u[j + i * width + k * height * width] = (aType)src.Get(i, j, k)/255.f;
				u_bar[j + i * width + k * height * width] = (aType)src.Get(i, j, k)/255.f;
				p_x[j + i * width + k * height * width] = 0.f;
				p_y[j + i * width + k * height * width] = 0.f;
			}
		}
	}
}

template<typename aType>
void RealTimeMinimizer<aType>::DualAscent(aType* p_x, aType* p_y, aType* u_bar, aType sigma, aType lambda, aType nu) {
	int I;
	aType fac = (2.f * lambda) / (sigma + 2.f * lambda);
	aType B = (nu / lambda) * sigma * (sigma + 2.f * lambda);
	aType u1, u2, norm;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				I = j + i * width + k * height * width;
				u1 = i+1 < height ? (u_bar[j + (i+1) * width + k * height * width] - u_bar[I]) : 0.f;
				u2 = j+1 < width ? (u_bar[(j+1) + i * width + k * height * width] - u_bar[I]) : 0.f;
				u1 = p_x[I] + sigma * u1;
				u2 = p_y[I] + sigma * u2;
				norm = u1*u1+u2*u2;
				p_x[I] = norm <= B ? fac * u1 : 0.f;
				p_y[I] = norm <= B ? fac * u2 : 0.f;
			}
		}
	}
}

template<typename aType>
aType RealTimeMinimizer<aType>::PrimalDescent(aType* u_bar, aType* u, aType* p_x, aType* p_y, aType* f, aType tau, aType theta) {
	int I;
	aType u1, u2, un, tmp, norm = 0.f;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				I = j + i * width + k * height * width;
				u1 = (i+1 < height ? p_x[I] : 0.f) - (i>0 ? p_x[j + (i-1) * width + k * height * width] : 0.f);
				u2 = (j+1 < width ? p_y[I] : 0.f) - (j>0 ? p_y[(j-1) + i * width + k * height * width] : 0.f);
				un = u[I];
				tmp = un + tau * (u1+u2);
				u[I] = (tmp + 2.f * tau * f[I]) / (1.f + 2.f * tau);
				u_bar[I] = u[I] + theta * (u[I] - un);
				norm += fabs(u[I] - un);
			}
		}
	}
	return (norm/(height*width));
}

template<typename aType>
void RealTimeMinimizer<aType>::EdgeHighlighting(aType* u, aType lambda, aType nu) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int X = j + i * width + k * height * width;
				aType dx = i + 1 < height ? u[j + (i+1) * width + k * height * width] - u[X] : 0;
				aType dy = j + 1 < width ? u[j + 1 + i * width + k * height * width] - u[X] : 0;
				aType norm = sqrt(dx*dx + dy*dy);
				aType factor = 0.03f;
				if (norm >= factor) {
					aType c = (aType)1 / log(sqrt(2.f)/factor);
					u[X] = (1 - c * log(norm / factor)) * u[X];
				}
			}
		}
	}
}

template<typename aType>
void RealTimeMinimizer<aType>::SetSolution(Image<aType>& dst) {
	for (int k = 0; k < channel; k++)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				dst.Set(i, j, k, (aType)u[j + i * width + k * height * width]*255.f);
}

template<typename aType>
void RealTimeMinimizer<aType>::RTMinimizer(Image<aType>& src, Image<aType>& dst, aType lambda, aType nu, bool eh) {
	int k;
	aType nrj, nrjn = 0.f;
	aType tau = 0.25f;
	aType sigma = 0.5f;
	aType theta = 1.f;
	dst.Reset(height, width, channel, src.Type());
	Initialize(src);
	for (k = 1; k < steps; k++) {
		DualAscent(p_x, p_y, u_bar, sigma, lambda, nu);
		theta = 1.f / sqrt(1.f + 4.f * tau);
		nrj = PrimalDescent(u_bar, u, p_x, p_y, f, tau, theta);
		tau *= theta; sigma /= theta;
		if (!eh && fabs(nrj - nrjn) <= 5*1e-5) break;
		else nrjn = nrj;
	}
	if (eh) {
		EdgeHighlighting(u, lambda, nu);
	}
	cout << "Iterations: " << k << endl;
	SetSolution(dst);
}