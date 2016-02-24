template<typename aType>
RealTimeMinimizer<aType>::RealTimeMinimizer(Image<aType>& src, int steps) {
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
RealTimeMinimizer<aType>::~RealTimeMinimizer() {
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
void RealTimeMinimizer<aType>::Initialize(Image<aType>& src) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				f[j + i * width + k * height * width] = (aType)src.Get(i, j, k) / 255.f;
				u[j + i * width + k * height * width] = (aType)src.Get(i, j, k) / 255.f;
				u_n[j + i * width + k * height * width] = (aType)src.Get(i, j, k) / 255.f;
				u_bar[j + i * width + k * height * width] = (aType)src.Get(i, j, k) / 255.f;
				p_x[j + i * width + k * height * width] = 0.0;
				p_y[j + i * width + k * height * width] = 0.0;
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
void RealTimeMinimizer<aType>::Nabla(aType* gradient_x, aType* gradient_y, aType* u_bar, aType* p_x, aType* p_y, aType sigma) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				gradient_x[j + i * width + k * height * width] = i + 1 < height ? (u_bar[j + (i+1) * width + k * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
				gradient_y[j + i * width + k * height * width] = j + 1 < width ? (u_bar[j + 1 + i * width + k * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
				gradient_x[j + i * width + k * height * width] = sigma * gradient_x[j + i * width + k * height * width] + p_x[j + i * width + k * height * width];
				gradient_y[j + i * width + k * height * width] = sigma * gradient_y[j + i * width + k * height * width] + p_y[j + i * width + k * height * width];
			}
		}
	}
}

template<typename aType>
void RealTimeMinimizer<aType>::VectorOfInnerProduct(aType* vector_norm_squared, aType* p_tilde_x, aType* p_tilde_y) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < channel; k++) {
				vector_norm_squared[j + i * width] += (pow(p_tilde_x[j + i * width + k * height * width], 2) + pow(p_tilde_y[j + i * width + k * height * width], 2));
			}
			vector_norm_squared[j + i * width] = sqrt(vector_norm_squared[j + i * width]);
		}
	}
}

template<typename aType>
void RealTimeMinimizer<aType>::ProxRstar(aType* p_x, aType* p_y, aType* p_tilde_x, aType* p_tilde_y, aType alpha, aType lambda, aType sigma) {
	aType* vector_norm_squared = (aType*)malloc(height*width*sizeof(aType));
	aType factor = (2 * alpha) / (sigma + 2 * alpha);
	aType bound = sqrt((lambda / alpha) * sigma * (sigma + 2 * alpha));
	VectorOfInnerProduct(vector_norm_squared, p_tilde_x, p_tilde_y);
	// aType norm = 0.f;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int ix = j + i * width + k * height * width;
				int X = j + i * width;
				// norm = sqrt(p_tilde_x[ix]*p_tilde_x[ix] + p_tilde_y[ix]*p_tilde_y[ix]);
				// p_x[ix] = norm <= bound ? factor * p_tilde_x[ix] : 0;
				// p_y[ix] = norm <= bound ? factor * p_tilde_y[ix] : 0;
				p_x[ix] = vector_norm_squared[X] <= bound ? factor * p_tilde_x[ix] : 0;
				p_y[ix] = vector_norm_squared[X] <= bound ? factor * p_tilde_y[ix] : 0;
			}
		}
	}
	free(vector_norm_squared);
}

template<typename aType>
void RealTimeMinimizer<aType>::NablaTranspose(aType* gradient_transpose, aType* p_x, aType* p_y, aType* u_n, aType tau) {
	aType x = 0.0;
	aType x_minus_one = 0.0;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				x = i + 1 < height ? p_x[j + i * width + k * height * width] : 0.0;
				x_minus_one = i > 0 ? p_x[j + (i-1) * width + k * height * width] : 0.0;
				gradient_transpose[j + i * width + k * height * width] = x_minus_one - x;
				x = j + 1 < width ? p_y[j + i * width + k * height * width] : 0.0;
				x_minus_one = j > 0 ? p_y[j - 1 + i * width + k * height * width] : 0.0;
				gradient_transpose[j + i * width + k * height * width] += (x_minus_one - x);
				gradient_transpose[j + i * width + k * height * width] = u_n[j + i * width + k * height * width] - tau * gradient_transpose[j + i * width + k * height * width];
			}
		}
	}
}

template<typename aType>
void RealTimeMinimizer<aType>::ProxD(aType* u, aType* u_tilde, aType* f, aType tau) {
	for (int i = 0; i < size; i++)
		u[i] = (u_tilde[i] + 2.0 * tau * f[i]) / (1.0 + 2.0 * tau);
}

template<typename aType>
void RealTimeMinimizer<aType>::ProxDgamma(aType* u, aType* u_tilde, aType* u_n, aType* f, aType gamma, aType tau) {
	// aType q = 2.f;
	// aType taubar;
	// aType t;
	// aType u0;
	// aType uprev;
	for (int i = 0; i < size; i++)
	{
		// if (q == 1) {
			// aType fac = u_tilde[i] + 2.f*tau*f[i] - u_n[i]*(1 + 2*tau);
			// if (fac > gamma*tau)		u[i] = (u_tilde[i] + 2.f*tau*f[i] + 2*tau*gamma) / (1.f + 2*tau);
			// if (fac < -tau*gamma) 		u[i] = (u_tilde[i] + 2.f*tau*f[i] - 2*tau*gamma) / (1.f + 2*tau);
			// if (fabs(fac) <= tau*gamma) u[i] = u_n[i];
		// }
		u[i] = (u_tilde[i] + 2.f*tau*f[i] + 2*tau*gamma*u_n[i]) / (1.f + 2*tau + 2*tau*gamma);
		// if (q == 2) u[i] = (u_tilde[i] + 2.f*tau*f[i] + 2*tau*gamma*u_n[i]) / (1.f + 2*tau + 2*tau*gamma);
		// u0 = (u_tilde[i] + 2.0 * tau * f[i]) / (1.0 + 2.0 * tau);
		// uprev = u_n[i];
		// taubar = ((gamma*tau)/(1 + 2*tau)) * pow(fabs(u0 - uprev), q - 2);
		// t = (aType)1 / pow((3*taubar)/(aType)4 + sqrt(1 + pow((3*taubar)/(aType)4, 2)), 2);
		// t = (aType) 1 / (1 + 2 * taubar);
		// t = fmax(0.f, 1 - taubar);
		// u[i] = (1 - t)*uprev + t*u0;
	}
}

template<typename aType>
void RealTimeMinimizer<aType>::Extrapolation(aType* u_bar, aType* u, aType* u_n, aType theta) {
	for (int i = 0; i < size; i++)
	{
		u_bar[i] = u[i] + theta * (u[i] - u_n[i]);
		u_n[i] = u[i];
	}
}

template<typename aType>
aType RealTimeMinimizer<aType>::PrimalEnergy(aType* u, aType* g, aType alpha, aType lambda) {
	aType energy = 0;
	aType dx = 0;
	aType dy = 0;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int X = j + i * width + k * height * width;
				dx = i + 1 < height ? u[j + (i+1) * width + k * height * width] - u[X] : 0;
				dy = j + 1 < width ? u[j + 1 + i * width + k * height * width] - u[X] : 0;
				energy += min(alpha * (dx*dx + dy*dy), lambda);
				energy += (pow(u[X] - g[X], 2));
			}
		}
	}
	return energy;
}

template<typename aType>
aType RealTimeMinimizer<aType>::StopCriterion(aType* u, aType* u_n) {
	aType norm = 0;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int ix = j + i * width + k * height * width;
				norm += fabs(u[ix] - u_n[ix]);
			}
		}
	}
	return norm;
}

template<typename aType>
void RealTimeMinimizer<aType>::EdgeHighlighting(aType* u, aType alpha, aType lambda) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int X = j + i * width + k * height * width;
				aType dx = i + 1 < height ? u[j + (i+1) * width + k * height * width] - u[X] : 0;
				aType dy = j + 1 < width ? u[j + 1 + i * width + k * height * width] - u[X] : 0;
				aType norm = sqrt(dx*dx + dy*dy);
				aType factor = 0.03;
				// aType factor = sqrt(lambda/alpha);
				if (norm >= factor) {
					aType c = (aType)1 / log(2.f/factor);
					u[X] = (1 - c * log(norm / factor)) * u[X];
				}
			}
		}
	}
}

template<typename aType>
void RealTimeMinimizer<aType>::RTMinimizer(Image<aType>& src, Image<aType>& dst, aType alpha, aType lambda, aType gamma, aType tau) {
	int k;
	aType theta = 1;
	aType sigma = (aType)1 / (aType)(tau * 8);
	aType h = (aType)1 / (src.Height()*src.Width());
	aType stop;
	dst.Reset(height, width, channel, src.Type());
	Initialize(src);
	for (k = 1; k < steps; k++) {
		Nabla(gradient_x, gradient_y, u_bar, p_x, p_y, sigma);
		ProxRstar(p_x, p_y, gradient_x, gradient_y, alpha, lambda, sigma);
		NablaTranspose(gradient_transpose, p_x, p_y, u_n, tau);
		ProxD(u, gradient_transpose, f, tau);
		// if (gamma > 0) {
		// 	ProxD(u, gradient_transpose, f, tau);
		// 	// ProxDgamma(u, gradient_transpose, u_n, f, gamma, tau);
		// } else {
		// 	ProxD(u, gradient_transpose, f, tau);
		// }
		theta = (aType)1 / sqrt(1 + 4 * tau);
		tau *= theta;
		sigma /= theta;
		if (k%10 == 0) {
			stop = h * StopCriterion(u, u_n);
			if (stop < 5 * 1E-5) {
				break;
			}
		}
		Extrapolation(u_bar, u, u_n, theta);
	}
	if (gamma > 0) {
		EdgeHighlighting(u, alpha, lambda);
	}
	cout << "Iterations: " << k << endl;
	cout << "Estimated Primal Energy: " << PrimalEnergy(u, f, alpha, lambda) << endl;
	SetSolution(dst);
}