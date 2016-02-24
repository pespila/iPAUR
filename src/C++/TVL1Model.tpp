template<typename aType>
TVL1Model<aType>::TVL1Model(Image<aType>& src, int steps) {
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
TVL1Model<aType>::~TVL1Model() {
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
void TVL1Model<aType>::Initialize(Image<aType>& src) {
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
void TVL1Model<aType>::SetSolution(Image<aType>& dst) {
	for (int k = 0; k < channel; k++)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				dst.Set(i, j, k, (aType)u[j + i * width + k * height * width]);
}

template<typename aType>
void TVL1Model<aType>::Nabla(aType* gradient_x, aType* gradient_y, aType* u_bar, aType* p_x, aType* p_y, aType sigma) {
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
void TVL1Model<aType>::ProxRstar(aType* p_x, aType* p_y, aType* p_tilde_x, aType* p_tilde_y) {
	aType vector_norm = 0;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				vector_norm = sqrt(pow(p_tilde_x[j + i * width + k * height * width], 2) + pow(p_tilde_y[j + i * width + k * height * width], 2));
				p_x[j + i * width + k * height * width] = p_tilde_x[j + i * width + k * height * width] / fmax(1, vector_norm);
				p_y[j + i * width + k * height * width] = p_tilde_y[j + i * width + k * height * width] / fmax(1, vector_norm);
			}
		}
	}
}

template<typename aType>
void TVL1Model<aType>::NablaTranspose(aType* gradient_transpose, aType* p_x, aType* p_y, aType* u_n, aType tau) {
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
void TVL1Model<aType>::ProxD(aType* u, aType* u_tilde, aType* f, aType tau, aType lambda) {
	aType tau_mult_lambda = tau * lambda;
	aType u_tilde_minus_original_image = 0;
	for (int i = 0; i < size; i++) {
		u_tilde_minus_original_image = u_tilde[i] - f[i];
		if (u_tilde_minus_original_image > tau_mult_lambda) 			u[i] = u_tilde[i] - tau_mult_lambda;
		if (u_tilde_minus_original_image < -tau_mult_lambda) 			u[i] = u_tilde[i] + tau_mult_lambda;
		if (fabs(u_tilde_minus_original_image) <= tau_mult_lambda) 		u[i] = f[i];
	}
}

template<typename aType>
void TVL1Model<aType>::Extrapolation(aType* u_bar, aType* u, aType* u_n, aType theta) {
	for (int i = 0; i < size; i++)
	{
		u_bar[i] = u[i] + theta * (u[i] - u_n[i]);
		u_n[i] = u[i];
	}
}

template<typename aType>
aType TVL1Model<aType>::Energy(aType* u, aType* g, aType* p_x, aType* p_y, aType lambda) {
	aType energy = 0;
	aType dx = 0;
	aType dy = 0;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int X = j + i * width + k * height * width;
				dx = i + 1 < height ? u[j + (i+1) * width + k * height * width] - u[X] : 0;
				dy = j + 1 < width ? u[j + 1 + i * width + k * height * width] - u[X] : 0;
				energy += (dx * p_x[X] + dy * p_y[X]);
				energy += (lambda * (abs(u[X] - g[X])));
			}
		}
	}
	return energy;
}

template<typename aType>
aType TVL1Model<aType>::PrimalEnergy(aType* u, aType* g, aType lambda) {
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
				energy += (lambda * (abs(u[X] - g[X])));
			}
		}
	}
	return energy;
}

template<typename aType>
void TVL1Model<aType>::TVL1(Image<aType>& src, Image<aType>& dst, aType lambda, aType tau) {
	int k;
	aType theta = 1;
	aType sigma = (aType)1 / (aType)(tau * 8);
	aType energy = Energy(u, f, p_x, p_y, lambda);
	dst.Reset(height, width, channel, src.Type());
	Initialize(src);
	for (k = 1; k < steps; k++) {
		Nabla(gradient_x, gradient_y, u_bar, p_x, p_y, sigma);
		ProxRstar(p_x, p_y, gradient_x, gradient_y);
		NablaTranspose(gradient_transpose, p_x, p_y, u_n, tau);
		ProxD(u, gradient_transpose, f, tau, lambda);
		Extrapolation(u_bar, u, u_n, theta);
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