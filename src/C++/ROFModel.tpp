template<typename aType>
ROFModel<aType>::ROFModel(Image<aType>& src, int steps) {
	this->steps = steps;
	this->channel = src.Channels();
	this->height = src.Height();
	this->width = src.Width();
	this->size = height * width * channel;
	this->f = (aType*)malloc(size*sizeof(aType));
	this->u = (aType*)malloc(size*sizeof(aType));
	this->u_n = (aType*)malloc(size*sizeof(aType));
	this->u_bar = (aType*)malloc(size*sizeof(aType));
	this->gradient_x = (aType*)malloc(size*sizeof(aType));
	this->gradient_y = (aType*)malloc(size*sizeof(aType));
	this->gradient_transpose = (aType*)malloc(size*sizeof(aType));
	this->p_x = (aType*)malloc(size*sizeof(aType));
	this->p_y = (aType*)malloc(size*sizeof(aType));
}

template<typename aType>
ROFModel<aType>::~ROFModel() {
	free(f);
	free(u);
	free(u_n);
	free(u_bar);
	free(gradient_x);
	free(gradient_y);
	free(gradient_transpose);
	free(p_x);
	free(p_y);
}

template<typename aType>
void ROFModel<aType>::Initialize(Image<aType>& src) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				f[j + i * width + k * height * width] = (aType)src.Get(i, j, k);
				u[j + i * width + k * height * width] = (aType)src.Get(i, j, k);
				u_n[j + i * width + k * height * width] = (aType)src.Get(i, j, k);
				u_bar[j + i * width + k * height * width] = (aType)src.Get(i, j, k);
				p_x[j + i * width + k * height * width] = 0.0;
				p_y[j + i * width + k * height * width] = 0.0;
			}
		}
	}
}

template<typename aType>
void ROFModel<aType>::SetSolution(Image<aType>& dst) {
	for (int k = 0; k < channel; k++)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				dst.Set(i, j, k, (aType)u[j + i * width + k * height * width]);
}

template<typename aType>
void ROFModel<aType>::Nabla(aType* gradient_x, aType* gradient_y, aType* u_bar, aType* p_x, aType* p_y, aType sigma) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				gradient_x[j + i * width + k * height * width] = i + 1 < height ? u_bar[j + (i+1) * width + k * height * width] - u_bar[j + i * width + k * height * width] : 0.0;
				gradient_y[j + i * width + k * height * width] = j + 1 < width ? u_bar[j + 1 + i * width + k * height * width] - u_bar[j + i * width + k * height * width] : 0.0;
				gradient_x[j + i * width + k * height * width] = sigma * gradient_x[j + i * width + k * height * width] + p_x[j + i * width + k * height * width];
				gradient_y[j + i * width + k * height * width] = sigma * gradient_y[j + i * width + k * height * width] + p_y[j + i * width + k * height * width];
			}
		}
	}
}

template<typename aType>
void ROFModel<aType>::ProxRstar(aType* p_x, aType* p_y, aType* p_tilde_x, aType* p_tilde_y) {
	aType vector_norm = 0.0;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				vector_norm = sqrt(pow(p_tilde_x[j + i * width + k * height * width], 2) + pow(p_tilde_y[j + i * width + k * height * width], 2));
				p_x[j + i * width + k * height * width] = p_tilde_x[j + i * width + k * height * width] / fmax(1.0, vector_norm);
				p_y[j + i * width + k * height * width] = p_tilde_y[j + i * width + k * height * width] / fmax(1.0, vector_norm);
			}
		}
	}
}

template<typename aType>
void ROFModel<aType>::NablaTranspose(aType* gradient_transpose, aType* p_x, aType* p_y, aType* u_n, aType tau) {
	aType x = 0.0;
	aType x_minus_one = 0.0;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				x = i + 1 < height ? p_x[j + i * width + k * height * width] : 0.0;
				x_minus_one = i > 0 ? p_x[j + (i-1) * width + k * height * width] : 0.0;
				gradient_transpose[j + i * width + k * height * width] = x - x_minus_one;
				x = j + 1 < width ? p_y[j + i * width + k * height * width] : 0.0;
				x_minus_one = j > 0 ? p_y[j - 1 + i * width + k * height * width] : 0.0;
				gradient_transpose[j + i * width + k * height * width] += x - x_minus_one;
				gradient_transpose[j + i * width + k * height * width] = u_n[j + i * width + k * height * width] + tau * gradient_transpose[j + i * width + k * height * width];
			}
		}
	}
}

template<typename aType>
void ROFModel<aType>::ProxD(aType* u, aType* u_tilde, aType* f, aType tau, aType lambda) {
	for (int i = 0; i < size; i++)
		u[i] = (u_tilde[i] + tau * lambda * f[i]) / (1.0 + tau * lambda);
}

template<typename aType>
void ROFModel<aType>::Extrapolation(aType* u_bar, aType* u, aType* u_n, aType theta) {
	for (int i = 0; i < size; i++)
	{
		u_bar[i] = u[i] + theta * (u[i] - u_n[i]);
		u_n[i] = u[i];
	}
}

template<typename aType>
aType ROFModel<aType>::Energy(aType* u, aType* g, aType* p_x, aType* p_y, aType lambda) {
	aType energy = 0;
	aType dx = 0;
	aType dy = 0;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int X = j + i * width + k * height * width;
				dx = i + 1 < height ? u[j + (i+1) * width + k * height * width] - u[X] : 0;
				dy = j + 1 < width ? u[j + 1 + i * width + k * height * width] - u[X] : 0;
				dx *= p_x[X];
				dy *= p_y[X];
				energy += (dx + dy);
				energy += (lambda/2 * pow(u[X] - g[X], 2));
			}
		}
	}
	return energy;
}

template<typename aType>
aType ROFModel<aType>::PrimalEnergy(aType* u, aType* g, aType lambda) {
	aType energy = 0;
	aType dx = 0;
	aType dy = 0;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int X = j + i * width + k * height * width;
				dx = i + 1 < height ? u[j + (i+1) * width + k * height * width] - u[X] : 0;
				dy = j + 1 < width ? u[j + 1 + i * width + k * height * width] - u[X] : 0;
				energy += sqrt(dx*dx + dy*dy);
				energy += (lambda/2 * pow(u[X] - g[X], 2));
			}
		}
	}
	return energy;
}

template<typename aType>
void ROFModel<aType>::DualAscent(aType* p_x, aType* p_y, aType* u_bar, aType sigma) {
	int I;
	aType u1, u2, norm;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				I = j + i * width + k * height * width;
				u1 = i+1 < height ? (u_bar[j + (i+1) * width + k * height * width] - u_bar[I]) : 0.0;
				u2 = j+1 < width ? (u_bar[(j+1) + i * width + k * height * width] - u_bar[I]) : 0.0;
				u1 = p_x[I] + sigma * u1;
				u2 = p_y[I] + sigma * u2;
				norm = sqrt(u1*u1+u2*u2);
				p_x[I] = u1/fmax(1.f, norm);
				p_y[I] = u2/fmax(1.f, norm);
			}
		}
	}
}

template<typename aType>
void ROFModel<aType>::PrimalDescent(aType* u_bar, aType* u, aType* p_x, aType* p_y, aType lambda, aType tau, aType theta) {
	int I;
	aType u1, u2, un;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				I = j + i * width + k * height * width;
				u1 = (i+1 < height ? p_x[I] : 0.0) - (i>0 ? p_x[j + (i-1) * width + k * height * width] : 0.0);
				u2 = (j+1 < width ? p_y[I] : 0.0) - (j>0 ? p_y[(j-1) + i * width + k * height * width] : 0.0);
				un = u[I];
				u[I] = ((un + tau * (u1+u2)) + tau * lambda * f[I]) / (1.0 + tau * lambda);
				u_bar[I] = u[I] + theta * (u[I] - un);
			}
		}
	}
}

template<typename aType>
void ROFModel<aType>::ROF(Image<aType>& src, Image<aType>& dst, aType lambda, aType tau) {
	int k;
	aType theta = 1.f;
	aType sigma = (aType)1 / (aType)(tau * 8);
	aType energy = Energy(u, f, p_x, p_y, lambda);
	dst.Reset(height, width, channel, src.Type());
	Initialize(src);
	for (k = 1; k < steps; k++) {
		DualAscent(p_x, p_y, u_bar, sigma);
		PrimalDescent(u_bar, u, p_x, p_y, lambda, tau, theta);
		// Nabla(gradient_x, gradient_y, u_bar, p_x, p_y, sigma);
		// ProxRstar(p_x, p_y, gradient_x, gradient_y);
		// NablaTranspose(gradient_transpose, p_x, p_y, u_n, tau);
		// ProxD(u, gradient_transpose, f, tau, lambda);
		// Extrapolation(u_bar, u, u_n, theta);
		if (k%10 == 0) {
			aType energy_tmp = Energy(u, f, p_x, p_y, lambda);
			if (abs(energy - energy_tmp) < 1E-6) {
				break;
			} else {
				energy = energy_tmp;
			}
		}
	}
	cout << "Iterations: " << k << endl;
	cout << "Estimated PrimalEnergy: " << PrimalEnergy(u, f, lambda) << endl;
	cout << "Estimated Energy: " << Energy(u, f, p_x, p_y, lambda) << endl;
	SetSolution(dst);
}