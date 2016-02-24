template<typename aType>
ImageInpainting<aType>::ImageInpainting(Image<aType>& src, int steps) {
	this->steps = steps;
	this->channel = src.Channels();
	this->height = src.Height();
	this->width = src.Width();
	this->size = height * width * channel;
	this->hash_table = (int*)malloc(this->height*this->width*sizeof(int));
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
ImageInpainting<aType>::~ImageInpainting() {
	free(hash_table);
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
void ImageInpainting<aType>::Initialize(Image<aType>& src) {
	int small_size = height*width;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				f[j + i * width + k * height * width] = (aType)src.Get(i, j, k);
				u[j + i * width + k * height * width] = (aType)src.Get(i, j, k);
				u_bar[j + i * width + k * height * width] = (aType)src.Get(i, j, k);
				p_x[j + i * width + k * height * width] = 0.0;
				p_y[j + i * width + k * height * width] = 0.0;
			}
		}
	}

	for (int i = 0; i < small_size; i++)
		hash_table[i] = f[i + 0 * small_size] == 0 ? 0 : 1;
		// hash_table[i] = f[i + 0 * small_size] == 0 && f[i + 1 * small_size] == 0 && f[i + 2 * small_size] == 0 ? 0 : 1;
}

template<typename aType>
void ImageInpainting<aType>::SetSolution(Image<aType>& dst) {
	for (int k = 0; k < channel; k++)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				dst.Set(i, j, k, (aType)u[j + i * width + k * height * width]);
}

template<typename aType>
void ImageInpainting<aType>::Nabla(aType* gradient_x, aType* gradient_y, aType* u_bar, aType* p_x, aType* p_y, aType sigma) {
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
void ImageInpainting<aType>::ProxRstar(aType* p_x, aType* p_y, aType* p_tilde_x, aType* p_tilde_y) {
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
void ImageInpainting<aType>::NablaTranspose(aType* gradient_transpose, aType* p_x, aType* p_y, aType* u_n, aType tau) {
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
void ImageInpainting<aType>::ProxD(aType* u, aType* u_tilde, aType* f, int* hash_table, aType tau, aType lambda) {
	int small_size = height*width;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < small_size; i++) {
			u[i + k * small_size] = !hash_table[i] ? u_tilde[i + k * small_size] : (u_tilde[i + k * small_size] + tau * lambda * f[i + k * small_size]) / (1.0 + tau * lambda);
		}
	}
}

template<typename aType>
void ImageInpainting<aType>::Extrapolation(aType* u_bar, aType* u, aType* u_n, aType theta) {
	for (int i = 0; i < size; i++)
	{
		u_bar[i] = u[i] + theta * (u[i] - u_n[i]);
		u_n[i] = u[i];
	}
}

template<typename aType>
aType ImageInpainting<aType>::PrimalEnergy(aType* u, aType* g, int* hash_table, aType lambda) {
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
				energy += hash_table[i] ? (lambda/2 * pow(u[X] - g[X], 2)) : 0.f;
			}
		}
	}
	return energy;
}

template<typename aType>
void ImageInpainting<aType>::Inpaint(Image<aType>& src, Image<aType>& dst, aType lambda, aType tau) {
	int k;
	aType theta = 1;
	aType sigma = (aType)1 / (aType)(tau * 8);
	aType energy = PrimalEnergy(u, f, hash_table, lambda);
	dst.Reset(height, width, channel, src.Type());
	Initialize(src);
	for (k = 1; k < steps; k++)
	{
		Nabla(gradient_x, gradient_y, u_bar, p_x, p_y, sigma);
		ProxRstar(p_x, p_y, gradient_x, gradient_y);
		NablaTranspose(gradient_transpose, p_x, p_y, u_n, tau);
		ProxD(u, gradient_transpose, f, hash_table, tau, lambda);
		Extrapolation(u_bar, u, u_n, theta);
		if (k%10 == 0) {
			aType energy_tmp = PrimalEnergy(u, f, hash_table, lambda);
			if (abs(energy - energy_tmp) < 1E-6) {
				break;
			} else {
				energy = energy_tmp;
			}
		}
	}
	cout << "Iterations: " << k << endl;
	cout << "Estimated Primal Energy: " << PrimalEnergy(u, f, hash_table, lambda) << endl;
	SetSolution(dst);
}