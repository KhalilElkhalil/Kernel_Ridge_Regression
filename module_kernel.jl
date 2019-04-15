# Module for useful functions
# Copyright Khalil Elkhalil, KAUST 2018

using Distributions

# Artificial data generating function f(.)
function func(x, inv_Sigma_r, beta, bias=0)
	p, ~ = size(x);
	return (sin.(beta'*x))';
end

#solving te fixed point equation to obtain 1/n.tr(X'X/n-zI)^(-1): X = sqrtm(Cov)*Z
function stieltjes_equation(z, U, d, p, n, cov="general")
	# U :is the matrix of eigenvectos of Cov matrix
	# d :array of eigenvalues of Cov
	# Cov = U*diagm(d)*U';
	# z should be negative for the Stieltjes transform to be defined

	if cov == "general"
		c = p/n;
		error = 1;
		m_z = 1;
		while(error > 1e-6)
			arr = m_z*d + ones(p);
			arr = d.*(ones(p)./arr);
			aux = -1/(c*z-1/n*sum(arr));
			error = abs(m_z-aux)^2;
			m_z = aux;
		end
	elseif cov == "identity"
			m_z = (-(c*z-c+1) - sqrt((c*z-c-1)^2 -4*c)) /2/c/z;
	else
		println("check the covariance type");
    end

	# derivative of m_z
	m_z_prime = c / ( 1/m_z^2 - 1/n*sum(d.^2 ./(ones(p)+m_z*d).^2) );
	return m_z, m_z_prime;
end

#= In-sample empirical prediction risk
function risk_train_emp(lamda, K, A_c_lamda, f_X, num_real)
	n = (K.shape)[0];
	#t = np.random.randn(n)
	#f_X = np.matmul(sqrtm(K), t) # Bayesian model
	b = 0; # bias
	risk = 0;
	for i=1:num_real
		eps=randn(n);
		y=f_X+eps+b*ones(n)
		f_hat=eye(n)-lamda*A_c_lamda*P*y;
		risk=risk+1/n*norm(f_X-f_hat)^2;
    end

	return risk/num_real;
end
=#

function tuple_to_array(v, num_out)

  w = zeros(1, num_out);
  for i =1:length(v)
    temp = collect(v[i])' ;
    w = vcat(w, temp)
  end
  w = w[2:end, :];
  return w;
end

#=
function CKRR_test_error(U, d, p, n, Sigma, g0_prime, grad)
	nu = kernel(trace(Sigma)/p) - kernel(0) - g0_prime*trace(Sigma)/p;
	z = -(nu+lam)/g0_prime;
	ST = stieltjes_equation(z, U, d, p, n);
	m_z = ST[1];
	m_z_prime = ST[2];
	T_z = -1/z*U * diagm(ones(p)./(ones(p) + m_z*d)) * U';np.linspace(2.0, 3.0, num=5)
	tau = 1/p*trace(Sigma*T_z);
	phi = 1/p*trace(Sigma^2*T_z^2);
	delta = (tau+1)*phi/(z*m_z*phi+tau+1);
	C_i = var(ff_X)/c/(1+tau)^2*delta;
	C_ij = 1/c^2/(1+tau)^2 * grad'*Sigma*T_z*Sigma*T_z*Sigma*grad*(1+(1-tau)*delta*m_z)-2*delta/c^2/(1+tau)^3*grad'*Sigma*T_z*Sigma*grad;
	C = C_i + C_ij;
	A = C + var(ff_X);
	B = sig2/p*trace(Sigma*T_z) + sig2*z*inv(1-phi/c/(1+tau)) * 1/p*trace(Sigma*T_z^2);
	risk_test_rmt = A+B -2/c/(1+tau)*grad'*Sigma*(T_z)*Sigma*grad;

	return risk_test_rmt;
end
=#

# Limiting risks based on Theorem 1
function CKRR_risk(lam, U, d, varf, g0_prime, grad, nu, p, n, sig2, risk_type, cov="general")
	""" returns the asymptotic testing risk """
	c = p/n;
	z = -(nu+lam)/g0_prime;

	""" General covariance matrix """
	ST = stieltjes_equation(z, U, d, p, n);
	m_z = ST[1];
	#m_z_prime = ST[2];
	grad_U = U'*grad;
	tr_Sigma2T2 = sum(d.^2 ./(ones(p) + m_z*d).^2);

	if risk_type == "testing"
	# Testing risk
		risk_rmt = (n*varf + sig2*m_z^2*tr_Sigma2T2)/(n-m_z^2*tr_Sigma2T2) - n*m_z/(n-m_z^2*tr_Sigma2T2)*grad_U' * diagm(d.^2 ./ (ones(p) + m_z*d) + d.^2 ./ (ones(p) + m_z*d).^2) * grad_U;
	elseif risk_type == "training"

	# Training risk
		variance = sig2*(1-2*lam*c*m_z/g0_prime + lam^2/g0_prime^2*n*c^2*m_z^2/(n-m_z^2*tr_Sigma2T2));
		bias = lam^2/g0_prime^2*n*c^2*m_z^2/(n-m_z^2*tr_Sigma2T2) * (varf - m_z*grad_U' * diagm(d.^2 ./ (ones(p) + m_z*d) + d.^2 ./ (ones(p) + m_z*d).^2) * grad_U);
		risk_rmt = bias + variance;

	else
		println("risk type is wrong!")
	end

	""" Case where Sigma = I """
	if cov == "identity"
	m_z = (-(c*z-c+1) - sqrt((c*z-c-1)^2 -4*c)) /2/c/z;
	risk_rmt = (p*sig2*m_z^2 + n*varf*(1+m_z)^2 -n*m_z*(2+m_z)*norm(grad)^2) / (n*(1+m_z)^2 -p*m_z^2);
	end

	return risk_rmt
end

# Consistent estim. of the prediction risk based on Theorem 2 and Lemma 2
function CKRR_risk_estim(z_hat, Q_z_tilde, sig2, X, y, cov="general")
	# Computes a consistent estiamtor for the prediction risk
	p, n = size(X);
	c = p/n;
	#nu = kernel(trace(X*X'/n)/p) - kernel(0) - g0_prime*trace(X*X'/n)/p;
	#z = -(nu+lam)/g0_prime;
	z = z_hat;
	P = eye(n) - ones(n,n)/n;
	if cov == "identity"
		m_z_hat = (-(c*z-c+1) - sqrt((c*z-c-1)^2 -4*c)) /2/c/z;
		risk_hat = (n*var(y)*(1+m_z_hat)^2 -m_z_hat*(2+m_z_hat)*(mean(diag(y'*P*X'*X/n*P*y)) -p*var(y))) / (n*(1+m_z_hat)^2 -p*m_z_hat^2) - sig2;
	else
		#Q_z_tilde = inv(X*P*X'/p - z*eye(p));
		m_z_hat = 1/p*trace(Q_z_tilde) + (p-n)/p/z;
		risk_hat = mean(diag(1/(c*z*m_z_hat)^2*(1/n/p*y'*P*X'*(z*Q_z_tilde^2 - Q_z_tilde)*X*P*y))) + 1/(c*z*m_z_hat)^2*var(y) - sig2;
	end
	return risk_hat;
end

# Estimating the gradient (we didn't use this function in the main codes)
function grad_estim(S, inv_Sigma_r, beta, basis, eps=10.0^(-7))
	""" Body of the function """
	p, n = size(S);
	# We can process data in chunks: split n into k partsrisk_iter_rmt = CKRR_risk(lam, U, d, varf, g0_prime, grad, nu, p, n, sig2, "training");
	grad = (func(kron(ones(p)', S)+eps*kron(eye(p), ones(n)'), inv_Sigma_r, beta, basis) - func(kron(ones(p)', S), inv_Sigma_r, beta, basis)) ./ (eps*ones(n*p));
	grad = (ones(n)'/n*reshape(grad, n, p))';
	return grad;
end


""" Bernoulli dist.
srand(3);
global bern05 = Bernoulli(0.5);
Z = 2*rand(bern05, p, n) - ones(p, n);
global X = Sigma_r*Z; # correlated samples (centered)
"""
