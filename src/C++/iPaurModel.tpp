template<typename aType>
iPaurModel<aType>::iPaurModel(Image<aType>& src, int steps) {
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
iPaurModel<aType>::~iPaurModel() {
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
void iPaurModel<aType>::Initialize(Image<aType>& src) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				f[j + i * width + k * height * width] = (aType)src.Get(i, j, k);
				u[j + i * width + k * height * width] = (aType)src.Get(i, j, k);
				u_n[j + i * width + k * height * width] = (aType)src.Get(i, j, k);
				u_bar[j + i * width + k * height * width] = (aType)src.Get(i, j, k);
				p_x[j + i * width + k * height * width] = 0;
				p_y[j + i * width + k * height * width] = 0;
			}
		}
	}
}

template<typename aType>
void iPaurModel<aType>::SetSolution(Image<aType>& dst) {
	for (int k = 0; k < channel; k++)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				dst.Set(i, j, k, (aType)u[j + i * width + k * height * width]);
}

template<typename aType>
void iPaurModel<aType>::Nabla(aType* gradient_x, aType* gradient_y, aType* u_bar, aType* p_x, aType* p_y, aType sigma) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				gradient_x[j + i * width + k * height * width] = i + 1 < height ? u_bar[j + (i+1) * width + k * height * width] - u_bar[j + i * width + k * height * width] : 0;
				gradient_y[j + i * width + k * height * width] = j + 1 < width ? u_bar[j + 1 + i * width + k * height * width] - u_bar[j + i * width + k * height * width] : 0;
				gradient_x[j + i * width + k * height * width] = sigma * gradient_x[j + i * width + k * height * width] + p_x[j + i * width + k * height * width];
				gradient_y[j + i * width + k * height * width] = sigma * gradient_y[j + i * width + k * height * width] + p_y[j + i * width + k * height * width];
			}
		}
	}
}

template<typename aType>
void iPaurModel<aType>::VectorOfInnerProduct(aType* vector_norm_squared, aType* p_tilde_x, aType* p_tilde_y) {
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
void iPaurModel<aType>::ProxRstar(aType* p_x, aType* p_y, aType* p_tilde_x, aType* p_tilde_y, aType gamma, aType lambda, aType sigma) {
	aType* vector_norm_squared = (aType*)malloc(height*width*sizeof(aType));
	aType factor = (2 * gamma) / (sigma + 2 * gamma);
	aType bound = sqrt((lambda / gamma) * sigma * (sigma + 2 * gamma));
	VectorOfInnerProduct(vector_norm_squared, p_tilde_x, p_tilde_y);
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int ix = j + i * width + k * height * width;
				int X = j + i * width;
				p_x[ix] = vector_norm_squared[X] <= bound ? factor * p_tilde_x[ix] : 0;
				p_y[ix] = vector_norm_squared[X] <= bound ? factor * p_tilde_y[ix] : 0;
			}
		}
	}
	free(vector_norm_squared);
}

template<typename aType>
void iPaurModel<aType>::NablaTranspose(aType* gradient_transpose, aType* p_x, aType* p_y, aType* u_n, aType tau) {
	aType x = 0;
	aType x_minus_one = 0;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				x = i + 1 < height ? p_x[j + i * width + k * height * width] : 0;
				x_minus_one = i > 0 ? p_x[j + (i-1) * width + k * height * width] : 0;
				gradient_transpose[j + i * width + k * height * width] = x_minus_one - x;
				x = j + 1 < width ? p_y[j + i * width + k * height * width] : 0;
				x_minus_one = j > 0 ? p_y[j - 1 + i * width + k * height * width] : 0;
				gradient_transpose[j + i * width + k * height * width] += x_minus_one - x;
				gradient_transpose[j + i * width + k * height * width] = u_n[j + i * width + k * height * width] - tau * gradient_transpose[j + i * width + k * height * width];
			}
		}
	}
}

template<typename aType>
void iPaurModel<aType>::ProxD(aType* u, aType* u_tilde, aType* f, aType tau, aType alpha, aType beta) {
	aType tau_alpha = tau * alpha;
	for (int i = 0; i < size; i++) {
		if (u_tilde[i] - f[i] > tau_alpha) 				u[i] = (u_tilde[i] - tau_alpha + tau * beta * f[i])/(1 + tau * beta);
		if (u_tilde[i] - f[i] < -tau_alpha) 			u[i] = (u_tilde[i] + tau_alpha + tau * beta * f[i])/(1 + tau * beta);
		if (fabs(u_tilde[i] - f[i]) <= tau * alpha) 	u[i] = f[i];
	}
}

template<typename aType>
void iPaurModel<aType>::Extrapolation(aType* u_bar, aType* u, aType* u_n, aType theta) {
	for (int i = 0; i < size; i++)
	{
		u_bar[i] = u[i] + theta * (u[i] - u_n[i]);
		u_n[i] = u[i];
	}
}

template<typename aType>
aType iPaurModel<aType>::PrimalEnergy(aType* u, aType* g, aType alpha, aType beta) {
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
				energy += (alpha/2 * (abs(u[X] - g[X])));
				energy += (beta/2 * pow(u[X] - g[X], 2));
			}
		}
	}
	return energy;
}

template<typename aType>
aType iPaurModel<aType>::StopCriterion(aType* u, aType* u_n) {
	aType norm = 0;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int ix = j + i * width + k * height * width;
				norm += fabs(u[ix]/255.f - u_n[ix]/255.f);
			}
		}
	}
	return norm;
}

template<typename aType>
void iPaurModel<aType>::iPaur(Image<aType>& src, Image<aType>& dst, aType alpha, aType beta, aType gamma, aType lambda, aType tau) {
	int k;
	aType theta = 1;
	aType sigma = (aType)1 / (aType)(tau * 8);
	dst.Reset(height, width, channel, src.Type());
	aType h = (aType)1 / (src.Height()*src.Width());
	aType stop;
	Initialize(src);
	for (k = 1; k < steps; k++) {
		Nabla(gradient_x, gradient_y, u_bar, p_x, p_y, sigma);
		ProxRstar(p_x, p_y, gradient_x, gradient_y, gamma, lambda, sigma);
		NablaTranspose(gradient_transpose, p_x, p_y, u_n, tau);
		ProxD(u, gradient_transpose, f, tau, alpha, beta);
		if (k > 250 && k%10 == 0) {
			stop = h * StopCriterion(u, u_n);
			if (stop < 5 * 1E-5) {
				break;
			}
		}
		theta = (aType)1 / sqrt(1 + 4 * tau);
		tau *= theta;
		sigma /= theta;
		Extrapolation(u_bar, u, u_n, theta);
	}
	cout << "Iterations: " << k << endl;
	cout << "Estimated Primal Energy: " << PrimalEnergy(u, f, alpha, beta) << endl;
	SetSolution(dst);
}