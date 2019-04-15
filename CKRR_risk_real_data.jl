# We compare the empirical out of sample prediction risk with our derived asymptoic formulas

# Please add this package using Pkg.add("package_name")
@everywhere using Distances
@everywhere using BenchmarkTools
@everywhere using MAT
@everywhere using MTH229

# include "module_kernel.jl" containing all the needed functions
@everywhere include("module_kernel.jl");

# Kernel information
@everywhere kernel(x) = exp(x);
g0_prime = kernel'(0);

# Loading the Communities and Crime Data Set from UCI machine learning repository
# Please refer to "communities_data.py" to see how we preprocess the data
# Read the data as a dictionary and extract the useful information as follows
data_dict = matread("/home/elkhalk//Dropbox/Kernel_Ridge_Regession/Python_codes/communities_data.mat")
#data_dict = matread("residential_data.mat")
#data_dict = matread("Twitter_data.mat")
#data_dict = matread("BlogFeedback_data.mat")
#data_dict = matread("Cargo_data.mat")
#data_dict = matread("Facebook_data.mat")
#data_dict = matread("music_data.mat")
#data_dict = matread("Tic_data.mat")
#data_dict = matread("MSD_data.mat")
X = data_dict["X"];
output = data_dict["y"]
p, n_total = size(X);

# Training size
n = Int64(round(n_total*0.6));
T = n_total - n; # size of the testing data
c = p/n; # Data ratio

# Noise variance
sig2 = 0.05;

# Sample covariance matrix using all data samples
X = X - X*ones(n_total, n_total)/n_total; # Centering the data
Sigma = 1/n_total * X*X'; # Sample cov. matrix
d, U = eig(Sigma); # eigen decomposition
inv_Sigma = U*diagm(ones(p)./d)*U'; # inverse Cov.

# Projection matrix
P = eye(n) - 1/n*ones(n, n); # projection matrix

# nu as defined in Theorem 1
nu = kernel(trace(Sigma)/p) - kernel(0) - g0_prime*trace(Sigma)/p;

# regularization vector
lam_vect = linspace(0.1, 50, 100);

risk = map(1:length(lam_vect)) do ll

	# Regularization parameter
	lam = lam_vect[ll];
	z_hat = -(nu+lam)/g0_prime;

	# Averaging over the training as highlighted in the paper
	Iter_num = 1000; # Total noisen samples
	real_X = pmap(1:500) do xx
		println(ll, " ", xx)

			# Generate a random pattern for the training dat
			pattern = randperm(n_total);
			X_train = X[:, pattern[1:n]];
			y_train = output[pattern[1:n]];
			X_test = X[:, pattern[1+n:end]];
			y_test = output[pattern[1+n:end]];
			X_T = X_test;

			# inner-product Kernel matrix
			K = kernel.(X_train'*X_train/p);
			K_T = kernel.(X_T'*X_train/p);

			# Useful quantities to estimate the prediction risk
			Q_z_tilde = inv(X_train*P*X_train'/p - z_hat*eye(p));
			m_z_hat = 1/p*trace(Q_z_tilde) + (p-n)/p/z_hat;
			Kc_inv = inv(P*K*P + lam*eye(n));

			# Averaging over the noise
			ff_X = y_train;
			ff_X_T = y_test;
			noise = sqrt(sig2)*randn(n, Iter_num);
			y = y_train*ones(Iter_num)' + noise;

			# Error on the training
			f_CKRR_train = (eye(n) - lam*Kc_inv*P)*y; # estimate on the training
			error_train =  sum(diag((ff_X*ones(Iter_num)' - f_CKRR_train)'*(ff_X*ones(Iter_num)' - f_CKRR_train))) /n/Iter_num;

			# Error on the testing data
			f_CKRR_test = (K_T*P - ones(T)*ones(n)'/n*K*P)*P*Kc_inv*P*y + ones(T)*(y'*ones(n)/n)'; # estimate on the testing
			error_test = sum(diag((ff_X_T*ones(Iter_num)' - f_CKRR_test)'*(ff_X_T*ones(Iter_num)' - f_CKRR_test))) /T/Iter_num;
			error_test_estim = mean(CKRR_risk_estim(z_hat, Q_z_tilde, sig2, X_train, y, "__")); # consistent estim. Theorem2
			error_test_estim_bis = (c*lam*m_z_hat/g0_prime)^(-2) * error_train- sig2*(g0_prime/c/lam/m_z_hat-1)^2; # consistent estim. Lemma2

	return error_test, error_test_estim, error_test_estim_bis, error_train;
	end

	real_X = tuple_to_array(real_X, 4)
	return mean(real_X[:, 1]), mean(real_X[:, 2]), mean(real_X[:, 3]), mean(real_X[:, 4]);

end

risk = tuple_to_array(risk, 4);
risk_test = risk[:, 1]
risk_estim = risk[:, 2]
risk_estim_bis = risk[:, 3]
risk_train = risk[:, 4]


#rm("results_real.mat");
matwrite("results_real.mat", Dict(
				"risk_test" => risk_test, "risk_estim" => risk_estim,
									"risk_estim_bis" => risk_estim_bis, "risk_train" => risk_train;
		 ));
