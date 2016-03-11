template<typename aType>
ImageInpainting<aType>::ImageInpainting(Image<aType>& src, int steps) {
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
ImageInpainting<aType>::~ImageInpainting() {
	free(f);
	free(u);
	free(u_bar);
	free(p_x);
	free(p_y);
	free(ht);
}

template<typename aType>
void ImageInpainting<aType>::Initialize(Image<aType>& src) {
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
	if (channel == 1) {
		this->ht = (int*)malloc(this->height*this->width*sizeof(int));
		for (int i = 0; i < height*width; i++)
			ht[i] = f[i] == 0 ? 0 : 1;
	} else {
		this->ht = (int*)malloc(this->height*this->width*this->channel*sizeof(int));
		for (int i = 0; i < height*width; i++)
			ht[i] = (f[i] == 0 && f[i + height*width] == 0 && f[i + 2*height*width] == 0) ? 0 : 1;
	}
}

template<typename aType>
void ImageInpainting<aType>::DualAscent(aType* p_x, aType* p_y, aType* u_bar, aType sigma) {
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
aType ImageInpainting<aType>::PrimalDescent(aType* u_bar, aType* u, aType* p_x, aType* p_y, aType lambda, aType tau, aType theta) {
	int I;
	aType u1, u2, un, norm = 0.f;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				I = j + i * width + k * height * width;
				u1 = (i+1 < height ? p_x[I] : 0.0) - (i>0 ? p_x[j + (i-1) * width + k * height * width] : 0.0);
				u2 = (j+1 < width ? p_y[I] : 0.0) - (j>0 ? p_y[(j-1) + i * width + k * height * width] : 0.0);
				un = u[I];
				u[I] = !ht[I] ? (un + tau * (u1+u2)) : ((un + tau * (u1+u2)) + tau * lambda * f[I]) / (1.0 + tau * lambda);
				u_bar[I] = u[I] + theta * (u[I] - un);
				norm += fabs(u[I] - un);
			}
		}
	}
	return (norm/(height*width));
}

template<typename aType>
void ImageInpainting<aType>::SetSolution(Image<aType>& dst) {
	for (int k = 0; k < channel; k++)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				dst.Set(i, j, k, (aType)u[j + i * width + k * height * width]);
}

template<typename aType>
void ImageInpainting<aType>::Inpaint(Image<aType>& src, Image<aType>& dst, aType lambda, aType tau) {
	int k;
	aType nrj, nrjn = 0.f;
	aType sigma = (aType)1 / (aType)(tau * 8);
	dst.Reset(height, width, channel, src.Type());
	Initialize(src);
	for (k = 1; k < steps; k++) {
		DualAscent(p_x, p_y, u_bar, sigma);
		nrj = PrimalDescent(u_bar, u, p_x, p_y, lambda, tau, 1.f);
		if (fabs(nrj - nrjn) <= 1e-6) break;
		else nrjn = nrj;
	}
	cout << "Iterations: " << k << endl;
	SetSolution(dst);
}