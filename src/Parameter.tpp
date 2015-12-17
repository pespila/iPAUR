template<typename aType>
Parameter<aType>::Parameter(aType alpha, aType lambda, aType tau, aType sigma, aType theta, aType L, aType nu, int cartoon) {
	this->alpha = alpha;
	this->lambda = lambda;
	this->tau = tau;
	this->sigma = sigma;
	this->theta = theta;
	this->L = L;
	this->nu = nu;
	this->cartoon = cartoon;
}

template<typename aType>
Parameter<aType>::~Parameter(){}