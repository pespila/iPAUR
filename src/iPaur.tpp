template<typename aType>
iPAUR<aType>::iPAUR(Image<aType>& src, int steps) {
	this->steps = steps;
	this->channel = src.GetChannels();
	this->height = src.GetHeight();
	this->width = src.GetWidth();
	this->size = height * width * channel;
	this->g = (aType*)malloc(size*sizeof(aType));
	this->u = (aType*)malloc(size*sizeof(aType));
	this->u_n = (aType*)malloc(size*sizeof(aType));
	this->u_bar = (aType*)malloc(size*sizeof(aType));
	this->v = (aType*)malloc(size*sizeof(aType));
	this->v_n = (aType*)malloc(size*sizeof(aType));
	this->v_bar = (aType*)malloc(size*sizeof(aType));
	this->gradient_ux = (aType*)malloc(size*sizeof(aType));
	this->gradient_uy = (aType*)malloc(size*sizeof(aType));
	this->gradient_vx = (aType*)malloc(size*sizeof(aType));
	this->gradient_vy = (aType*)malloc(size*sizeof(aType));
	this->gradient_transpose_u = (aType*)malloc(size*sizeof(aType));
	this->gradient_transpose_v = (aType*)malloc(size*sizeof(aType));
	this->p_x = (aType*)malloc(size*sizeof(aType));
	this->p_y = (aType*)malloc(size*sizeof(aType));
	this->q_x = (aType*)malloc(size*sizeof(aType));
	this->q_y = (aType*)malloc(size*sizeof(aType));
}

template<typename aType>
iPAUR<aType>::~iPAUR() {
	free(g);
	free(u);
	free(u_n);
	free(u_bar);
	free(v);
	free(v_n);
	free(v_bar);
	free(gradient_ux);
	free(gradient_uy);
	free(gradient_vx);
	free(gradient_vy);
	free(gradient_transpose_u);
	free(gradient_transpose_v);
	free(p_x);
	free(p_y);
	free(q_x);
	free(q_y);
}

template<typename aType>
void iPAUR<aType>::Initialize(Image<aType>& src) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				g[j + i * width + k * height * width] = (aType)src.Get(i, j, k);
				u[j + i * width + k * height * width] = (aType)src.Get(i, j, k);
				v[j + i * width + k * height * width] = (aType)src.Get(i, j, k);
				u_bar[j + i * width + k * height * width] = (aType)src.Get(i, j, k);
				v_bar[j + i * width + k * height * width] = (aType)src.Get(i, j, k);
				p_x[j + i * width + k * height * width] = 0.0;
				p_y[j + i * width + k * height * width] = 0.0;
				q_x[j + i * width + k * height * width] = 0.0;
				q_y[j + i * width + k * height * width] = 0.0;
			}
		}
	}
}

template<typename aType>
void iPAUR<aType>::SetSolution(WriteableImage<aType>& dst) {
	for (int k = 0; k < channel; k++)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				dst.Set(i, j, k, (unsigned char)u[j + i * width + k * height * width]);
}

template<typename aType>
void iPAUR<aType>::Nabla(aType* gradient_x, aType* gradient_y, aType* u_bar) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				gradient_x[j + i * width + k * height * width] = i + 1 < height ? (u_bar[j + (i+1) * width + k * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
				gradient_y[j + i * width + k * height * width] = j + 1 < width ? (u_bar[j + 1 + i * width + k * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
			}
		}
	}
}

template<typename aType>
void iPAUR<aType>::ProxRstarU(aType* p_x, aType* p_y, aType* p_tilde_x, aType* p_tilde_y) {
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
void iPAUR<aType>::ProxRstarV(aType* p_x, aType* p_y, aType* p_tilde_x, aType* p_tilde_y) {
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
void iPAUR<aType>::NablaTranspose(aType* gradient_transpose, aType* p_x, aType* p_y) {
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
			}
		}
	}
}

template<typename aType>
void iPAUR<aType>::ProxDu(aType* u, aType* u_tilde, aType* g, aType* v, aType tau, aType alpha, aType beta) {
	for (int i = 0; i < size; i++) {
		aType factor = u_tilde[i] - (1.f - tau * alpha)*g[i];
		if (factor > tau*alpha*v[i] + tau*beta) 				u[i] = (u_tilde[i] - tau*beta - tau*alpha*v[i])/(1-tau*alpha);
		if (factor < tau*alpha*v[i] - tau*beta) 				u[i] = (u_tilde[i] + tau*beta - tau*alpha*v[i])/(1-tau*alpha);
		if (fabs(factor) < tau * (beta + alpha * v[i])) 		u[i] = g[i];
	}
}

template<typename aType>
// void iPAUR<aType>::ProxDv(aType* u, aType* u_tilde, aType* g, aType tau, aType alpha) {
// 	aType tau_mult_alpha = tau * alpha;
// 	aType u_tilde_minus_original_image = 0.0;
// 	for (int i = 0; i < size; i++) {
// 		u_tilde_minus_original_image = u_tilde[i] - g[i];
// 		if (u_tilde_minus_original_image > tau_mult_alpha) 				u[i] = u_tilde[i] - tau_mult_alpha;
// 		if (u_tilde_minus_original_image < -tau_mult_alpha) 			u[i] = u_tilde[i] + tau_mult_alpha;
// 		if (fabs(u_tilde_minus_original_image) <= tau_mult_alpha) 		u[i] = g[i];
// 	}
// }

template<typename aType>
void iPAUR<aType>::ProxDv(aType* u, aType* u_tilde, aType* g, aType tau, aType alpha) {
	for (int i = 0; i < size; i++)
		u[i] = (u_tilde[i] + tau * alpha * g[i]) / (1.0 + tau * alpha);
}

template<typename aType>
void iPAUR<aType>::iPAURmodel(Image<aType>& src, WriteableImage<aType>& dst, aType alpha, aType beta) {
	int i;
	dst.Reset(height, width, src.GetType());
	Initialize(src);
	aType theta = 1.f;
	aType L2 = 8;
	aType tau = 0.25;
	aType sigma = 1.f / (L2 * tau);
	int step = 10;
	for (int k = 0; k < steps; k++)
	{
		for (int l = 0; l < step; l++)
		{

		for (i = 0; i < size; i++) {v_n[i] = v[i];}
		Nabla(gradient_vx, gradient_vy, v_bar);
		for (i = 0; i < size; i++) {gradient_vx[i] = sigma * gradient_vx[i] + q_x[i];}
		for (i = 0; i < size; i++) {gradient_vy[i] = sigma * gradient_vy[i] + q_y[i];}
		ProxRstarV(q_x, q_y, gradient_vx, gradient_vy);
		NablaTranspose(gradient_transpose_v, q_x, q_y);
		for (i = 0; i < size; i++) {gradient_transpose_v[i] = v_n[i] - tau * gradient_transpose_v[i];}
		ProxDv(v, gradient_transpose_v, u, tau, alpha);
		for (i = 0; i < size; i++) {v_bar[i] = v[i] + theta * (v[i] - v_n[i]);}

		}

		for (int l = 0; l < 2*step; l++)
		{
			
		for (i = 0; i < size; i++) {u_n[i] = u[i];}
		Nabla(gradient_ux, gradient_uy, u_bar);
		for (i = 0; i < size; i++) {gradient_ux[i] = sigma * gradient_ux[i] + p_x[i];}
		for (i = 0; i < size; i++) {gradient_uy[i] = sigma * gradient_uy[i] + p_y[i];}
		ProxRstarU(p_x, p_y, gradient_ux, gradient_uy);
		NablaTranspose(gradient_transpose_u, p_x, p_y);
		for (i = 0; i < size; i++) {gradient_transpose_u[i] = u_n[i] - tau * gradient_transpose_u[i];}
		ProxDu(u, gradient_transpose_u, g, v, tau, alpha, beta);
		for (i = 0; i < size; i++) {u_bar[i] = u[i] + theta * (u[i] - u_n[i]);}
		
		}
		// for (i = 0; i < size; i++) {u_bar[i] = (u_bar[i] + v_bar[i]) / 2;}
		// for (i = 0; i < size; i++) {u[i] = tau * u[i] + (1-tau) * v[i];}
		// for (i = 0; i < size; i++) {u_bar[i] = u[i] + theta * (u[i] - v[i]);}
		// for (i = 0; i < size; i++) {u[i] = fabs(u[i]-v[i]);}
	}
	for (i = 0; i < size; i++) {u[i] = v[i];}
	// for (i = 0; i < size; i++) {u[i] = (fabs(u[i]) + fabs(v[i])) / 2.f;}
	SetSolution(dst);
}
