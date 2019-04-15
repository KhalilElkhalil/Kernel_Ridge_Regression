# We compare the empirical out of sample prediction risk with our derived asymptoic formulas

# Please add this package using Pkg.add("package_name")
@everywhere using Distances
@everywhere using BenchmarkTools
@everywhere using MAT
# @everywhere using MTH229

# include "module_kernel.jl" containing all the needed functions
@everywhere include("module_kernel.jl");

# Kernel information
@everywhere kernel(x) = exp(x);
g0_prime = 1;

# Regularizer grid
# lam_vect = linspace(0.3, 20, 100);
lam_vect = [2.887]
# problem's dimensions
p = 100; # Data dimension
n = Int64(round(2*p)); # training size
c = p/n; # data ratio

# Projection matrix
P = eye(n) - 1/n*ones(n, n);

# Noise variance
sig2 = 0.5;

# Data covariance matrix \Sigma
rho = 0.4; # correlation factor
v = 0:1:p-1;
Sigma = (rho*ones(p, p)) .^(abs.(v*ones(p)' - ones(p)*v')); # Sigma_{i,j} = rho^(abs(i-j))
# Sigma = eye(p);
d, U = eig(Sigma); # eigendecompositi
Sigma_r = U*diagm(sqrt.(d))*U'; # sqrtm(Sigma)
inv_Sigma = U*diagm(ones(p)./d)*U'; # Sigma^{-1}
inv_Sigma_r = U*diagm(ones(p)./ sqrt.(d))*U'; # Sigma^{-1/2};

# Statistics of the Output function: f(x) = sin(1^T.x/sqrt(p))
beta = ones(p)/sqrt(p);
beta_sigma_inner = beta'*Sigma*beta;
grad = beta*exp(-beta_sigma_inner/2);
varf = exp(-beta_sigma_inner)*sinh(beta_sigma_inner);

# nu as defined in Theorem1
nu = kernel(trace(Sigma)/p) - kernel(0) - g0_prime*trace(Sigma)/p;

# information on the Averaging parameters
T = 1000; # number of testing samples
Iter_num = 1000; # number of noise samples

# CKRR risk vs. lambda
risk = map(1:length(lam_vect)) do ll

    # regularization factor
    lam = lam_vect[ll];

    # z as defined in Theorem 1
    z = -(nu+lam)/g0_prime;

    # Realization over the training X
    real_X = map(1:1) do xx
        println(ll, " ", xx);
            T = 1000;

            # Generate training X
            #X = rand(p, n) * (sqrt(12)) -sqrt(12)/2; # uniform mean=0, var=1
            X = rand([-1,1], (p, n)) # Bernoulli with p=0.5
            # X = randn(p, n)
            X = Sigma_r * X;
            #X = Sigma_r*randn(p, n);

            # Generate testing data
            # X_T = rand(p, T) * (sqrt(12)) -sqrt(12)/2; # uniform mean=0, var=1
            X_T = rand([-1,1], (p, T)) # Bernoulli with p=0.5
            # X_T = randn(p, T)
            X_T = Sigma_r * X_T;
            #X_T = Sigma_r*randn(p, T);

            # Output function on train/test data
            ff_X = func(X, inv_Sigma_r, beta);
            ff_X_T = func(X_T, inv_Sigma_r, beta);

            # inner-product Kernel matrix
            K = kernel.(X'*X/p);
            Kc_inv = inv(P*K*P + lam*eye(n));
            K_T = kernel.(X_T'*X/p);

            # Useful quantities for the consistent estimator
            nu_hat = kernel(trace(X*X'/n)/p) - kernel(0) - g0_prime*trace(X*X'/n)/p;
            z_hat = -(nu_hat+lam)/g0_prime;
            Q_z_tilde = inv(X*P*X'/p - z_hat*eye(p));
            m_z_hat = 1/p*trace(Q_z_tilde) + (p-n)/p/z_hat;

            # Averaging over the noise
            noise = sqrt(sig2)*randn(n, Iter_num);
            y = ff_X*ones(Iter_num)' + noise;

            # Training risk (empirical)
            f_CKRR_train = (eye(n) - lam*Kc_inv*P)*y; # estimate on the training
            error_train =  sum(diag((ff_X*ones(Iter_num)' - f_CKRR_train)'*(ff_X*ones(Iter_num)' - f_CKRR_train))) /n/Iter_num;

            # Prediction risk (evaluated on by averaging the error on the testing samples)
            f_CKRR_test = (K_T*P - ones(T)*ones(n)'/n*K*P)*P*Kc_inv*P*y + ones(T)*(y'*ones(n)/n)'; # estimate on testing
            error_test = sum(diag((ff_X_T*ones(Iter_num)' - f_CKRR_test)'*(ff_X_T*ones(Iter_num)' - f_CKRR_test))) /T/Iter_num;

            # Consistent estiamtor of the prediction risk
            error_test_estim = mean(CKRR_risk_estim(z_hat, Q_z_tilde, sig2, X, y, "__")); # Theorem 2
            error_test_estim_bis = (c*lam*m_z_hat/g0_prime)^(-2) * error_train - sig2*(g0_prime/c/lam/m_z_hat-1)^2; # Lemma 2

            #error_train_rmt = (c*lam*m_z_hat/g0_prime)^2 * (error_test+sig2) + sig2 -2*sig2*c*lam*m_z_hat/g0_prime;
            # Error on the Testing
            #=
            errors =  error_test,
                   #sum(diag((ff_X_T*ones(Iter_num)' - f_KRR_test)'*(ff_X_T*ones(Iter_num)' - f_KRR_test))) /T/Iter_num,
                   error_train,
                   #sum(diag((ff_X*ones(Iter_num)' - f_KRR_train)'*(ff_X*ones(Iter_num)' - f_KRR_train))) /n/Iter_num,
                   CKRR_risk(lam, U, d, varf, g0_prime, grad, nu, p, n, sig2, "testing", "general"),
                   #error_test_rmt,
                   error_train_rmt;
            =#
            #returned = tuplejoin(returned, errors);

        #end
    return error_test, error_test_estim, error_test_estim_bis, error_train;
    end
    real_X = tuple_to_array(real_X, 4)

    # Limiting risk
    error_test_rmt = CKRR_risk(lam, U, d, varf, g0_prime, grad, nu, p, n, sig2, "testing", "general");
    error_train_rmt = CKRR_risk(lam, U, d, varf, g0_prime, grad, nu, p, n, sig2, "training", "general");

    return  mean(real_X[:, 1]), error_test_rmt, mean(real_X[:, 2]), mean(real_X[:, 3]), mean(real_X[:, 4]), error_train_rmt;
        #return (error_test - error_test_rmt)^2, (error_train - error_train_rmt)^2;
    #end
    #train_real = tuple_to_array(train_real, 2);
    #=
            """ information vector """
            S = Sigma_r * randn(p, test_num);
            k_S = kernel.(X'*S/p);
            k_0 = kernel.(X'*S);
            #k_c_S = P*k_S-1/n*P*(UK*diagm(dK)*UK')*ones(n, test_num);
            k_c_S = P*k_S-1/n*P*K*ones(n, test_num);
            #risk_test_krr = sig2*mean(diag(k_S' * K_inv *K_inv * k_S)) + norm(k_S'*K_inv*ff_X - func(S, inv_Sigma_r, beta, basis))^2/test_num;
            risk_test_krr = sig2*mean(diag(k_0' * K_inv *K_inv * k_S)) + norm(k_0'*K_inv*ff_X - func(S, inv_Sigma_r, beta, basis))^2/test_num;


            risk_test_emp = norm(k_c_S'*Kc_inv*P*ff_X + mean(ff_X)*ones(test_num) - func(S, inv_Sigma_r, beta, basis))^2/test_num  + sig2*mean(diag(k_c_S' * Kc_inv * P * Kc_inv * k_c_S));
            risk_train_emp = 1/n*norm(lam*Kc_train_real = tuple_to_array(train_real, 6);
            risk_train_asy = mean(train_real[:, 1]);
            risk_train_exp = mean(train_real[:, 2]);
            risk_test_asy = mean(train_real[:, 3]);
            risk_test_exp = mean(train_real[:, 4]);NTuple{0, Int}
            risk_train_krr = mean(train_real[:, 5]);
            rinv*P*ff_X)^2 + sig2/n*trace((eye(n)- lam*Kc_inv*P)^2);
            grad = grad_estim(X, inv_Sigma_r, beta, basis);
            #grad = inv_Sigma*X*diagm(ff_X)*ones(n)/n;
            #varf = var(y) - sig2;
            varf = var(ff_X);
            risk_train_krr = 1/n * trace((eye(n)-n*lam*K_inv)^2) + n*lam^2*norm(K_inv*ff_X)^2;
            risk_train_rmt = CKRR_risk(lam, U, d, varf, g0_prime, grad, nu, p, n, sig2, "training");
            risk_test_rmt = CKRR_risk(lam, U, d, varf, g0_prime, grad, nu, p, n, sig2, "testing");
            return risk_kerneltrain_rmt, risk_train_emp, risk_test_rmt, risk_test_emp, risk_train_krr, risk_test_krr;
            #return  (risk_iter_train - risk_iter_rmt)^2;
    =#
    ##end
    #=
            train_real = tuple_to_array(train_real, 6);
            risk_train_asy = mean(train_real[:, 1]);
            risk_train_exp = mean(train_real[:, 2]);
            risk_test_asy = mean(train_real[:, 3]);
            risk_test_exp = mean(train_real[:, 4]);
            risk_train_krr = mean(train_real[:, 5]);
            risk_test_krr = mean(train_real[:, 6]);
    =#
    #return risk_test_emp, risk_test_asy;
    """ MSE """
    #return (risk_test_emp - risk_test_rmt)^2;
    #return risk_train_asy, risk_train_exp, risk_test_asy, risk_test_exp, risk_train_krr, risk_test_krr;
    #risk = [ff_X_T f_CKRR_test f_KRR_test ff_X f_CKRR_train f_KRR_train]
    #return mean(train_real[:, 1]), mean(train_real[:, 2]);	tic();
end

risk = tuple_to_array(risk, 6);
risk_test = risk[:, 1];
risk_test_rmt = risk[:, 2];
risk_test_estim = risk[:, 3];
risk_test_estim_bis = risk[:, 4];
risk_train = risk[:, 5];
risk_train_rmt = risk[:, 6];

# rm("results.mat");
matwrite("results_synthetic1.mat", Dict(
                "risk_test" => risk_test, "risk_test_rmt" => risk_test_rmt,
                "risk_test_estim" => risk_test_estim, "risk_test_estim_bis" => risk_test_estim_bis,
                "risk_train" => risk_train, "risk_train_rmt" => risk_train_rmt;
         ));

#= Optimizing the CKRR testing risk
using Optim
optim_fun_CKRR_test(x) = CKRR_test_error(x[1], U, d, varf, g0_prime, grad, nu, p, n, sig2);
output_optim_CKRR_test = optimize(optim_fun_CKRR_test, [10,], BFGS()); # or GradientDescent()
lam_optim_CKRR_test = output_optim_CKRR_test.minimizer;
lam_optim_CKRR_test = lam_optim_CKRR_test[1];
minval_CKRR_test = output_optim_CKRR_test.minimum;rm
println("Optimal value = ", lam_optim_CKRR_test, ", Min value = ", minval_CKRR_test);
=#
##""" Saving in mat files """





#=
vars = matread("results.mat");
get!(vars, "sigmoid_test_emp", risk[:, 1])
get!(vars, "sigmoid_test_rmt", risk[:, 2])
get!(vars, "sigmoid_test_estim", risk[:, 3])
get!(vars, "sigmoid_test_estim_bis", risk[:, 4])
get!(vars, "sigmoid_train_emp", risk[:, 5])
get!(vars, "sigmoid_train_rmt", risk[:, 6])

matwrite("results.mat", vars);
=#




#=
using PyPlot
#rc('text', usetex=True)
#rc('font', family='serif')
plot(lam_vect, risk[:, 1], label=L"$\alpha + \beta$")
plot(lam_vect, risk[:, 2], "k--", label="Test RMT")
plot(lam_vect, risk[:, 3], label="Test Estim.")
plot(lam_vect, risk[:, 4], label="Test Estim. Bis.")
plot(lam_vect, risk[:, 5], label="Train Emp.")
plot(lam_vect, risk[:, 6], "k--", label="Limiting risk (Theorem 1)")
legend()
=#
#=
using MAT

if stat("results.mat").size == 0
    matwrite("results.mat", Dict(
                    "linear_test_emp" => risk[:, 1], "linear_test_rmt" => risk[:, 2],
                                        "linear_test_estim" => risk[:, 3], "linear_test_estim_bis" => risk[:, 4],
                                        "linear_train_emp" => risk[:, 5], "linear_train_rmt" => risk[:, 6];
             ));
    vars = matread("results.mat");
    # key(vars) is an iterator
elseif ("polynomial_test_emp" in keys(vars)) == false
    get!(vars, "polynomial_test_emp", risk[:, 1])
    get!(vars, "polynomial_test_rmt", risk[:, 2])
    get!(vars, "polynomial_test_estim", risk[:, 3])
    get!(vars, "polynomial_test_estim_bis", risk[:, 4])
    get!(vars, "polynomial_train_emp", risk[:, 5])
    get!(vars, "polynomial_train_rmt", risk[:, 6])
    #vars = matread("results.mat");

elseif ("sigmoid_test_emp" in keys(vars)) == false
    get!(vars, "sigmoid_test_emp", risk[:, 1])
    get!(vars, "sigmoid_test_rmt", risk[:, 2])
    get!(vars, "sigmoid_test_estim", risk[:, 3])
    get!(vars, "sigmoid_test_estim_bis", risk[:, 4])
    get!(vars, "sigmoid_train_emp", risk[:, 5])
    get!(vars, "sigmoid_train_rmt", risk[:, 6])
    #vars = matread("results.mat");

else
    get!(vars, "exponential_test_emp", risk[:, 1])
    get!(vars, "exponential_test_rmt", risk[:, 2])
    get!(vars, "exponential_test_estim", risk[:, 3])
    get!(vars, "exponential_test_estim_bis", risk[:, 4])
    get!(vars, "exponential_train_emp", risk[:, 5])
    get!(vars, "exponential_train_rmt", risk[:, 6])
    matwrite("results.mat", vars);
end
=#
