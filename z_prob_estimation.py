import numpy as np
# import pymc as pm
# import theano.tensor as tt
from scipy.linalg import inv


def baysian_mvg():
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    true_mu = np.random.randn(3)
    true_cov = np.diag(np.random.rand(3))
    data = np.random.multivariate_normal(true_mu, true_cov, size=n_samples)

    # Priors
    mu_0 = np.zeros(3)
    kappa_0 = 1.0
    nu_0 = 3 + 2  # degrees of freedom > number of dimensions
    Psi_0 = np.eye(3)

    with pm.Model() as model:
        # Inverse-Wishart prior for the covariance matrix
        Sigma = pm.WishartBartlett('Sigma', S=np.linalg.inv(Psi_0), nu=nu_0)
        
        # Normal prior for the mean
        mu = pm.MvNormal('mu', mu=mu_0, tau=kappa_0 * tt.inv(Sigma), shape=3)
        
        # Likelihood
        likelihood = pm.MvNormal('likelihood', mu=mu, cov=Sigma, observed=data)

        # Inference
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)

    # Posterior summary
    print(pm.summary(trace))

    # Posterior plots
    import arviz as az
    az.plot_trace(trace)
    az.plot_posterior(trace)





def mle(z):
    import numpy as np

    # Generate synthetic data
    # np.random.seed(42)
    # n_samples = 100
    # true_mu = np.random.randn(3)
    # true_cov = np.diag(np.random.rand(3))
    # data = np.random.multivariate_normal(true_mu, true_cov, size=n_samples)
    # print(z.shape)
    # print(data.shape)
    # print("Hello?")
    data = z
    # Maximum Likelihood Estimation
    def mle_multivariate_gaussian(data):
        n, d = data.shape
        mu_mle = np.mean(data, axis=0)
        sigma_mle = np.cov(data, rowvar=False, bias=True)  # `bias=True` uses normalization by n
        
        return mu_mle, sigma_mle

    mu_mle, sigma_mle = mle_multivariate_gaussian(data)

    print("MLE Mean:\n", mu_mle)
    print("MLE Covariance Matrix:\n", sigma_mle)
    return mu_mle, sigma_mle

def mahalanobis_distance(x, mu, sigma):
    if not sigma.shape[0] == sigma.shape[1] == len(mu) == len(x):
        raise ValueError("Dimensions of input do not match.")
    
    delta = x - mu
    inv_sigma = inv(sigma)  # Inverse of the covariance matrix
    distance_squared = np.dot(delta.T, np.dot(inv_sigma, delta))  # (x - mu)^T Sigma^-1 (x - mu)
    distance = np.sqrt(distance_squared)
    
    return distance

def nlog_likelihood(x, mu, sigma):
    """
    Calculate the negative log likelihood of observing a data point x from a
    multivariate Gaussian distribution with mean mu and covariance matrix sigma.
    
    Parameters:
    - x: A numpy array representing the data point.
    - mu: The mean vector of the distribution.
    - sigma: The covariance matrix of the distribution.
    
    Returns:
    - The log likelihood of x.
    """
    d = len(x)
    inv_sigma = np.linalg.inv(sigma)
    det_sigma = np.linalg.det(sigma)
    if det_sigma <= 0:
        raise ValueError("Covariance matrix must have a positive determinant.")
    
    term1 = -0.5 * d * np.log(2 * np.pi)
    term2 = -0.5 * np.log(det_sigma)
    delta = x - mu
    term3 = -0.5 * np.dot(delta.T, np.dot(inv_sigma, delta))
    
    log_likelihood = term1 + term2 + term3
    return -log_likelihood


if __name__ == "__main__":
    # baysian_mvg()
    z = np.load("./z_embeddings.npy")
    print(z)
    
    mu_mle, sigma_mle = mle(z)
    
    # Example usage
    print(nlog_likelihood(z[0], mu_mle, sigma_mle))
    print(nlog_likelihood(z[1], mu_mle, sigma_mle))
    print(nlog_likelihood(z[2], mu_mle, sigma_mle))
    print(nlog_likelihood([0]*16, mu_mle, sigma_mle))
    # print("Log likelihood:", log_likelihood_value)

    
    # print(mahalanobis_distance(z[0], mu_mle, sigma_mle))
    # print(mahalanobis_distance(z[1], mu_mle, sigma_mle))
    # print(mahalanobis_distance(z[2], mu_mle, sigma_mle))
    # print(mahalanobis_distance([0]*16, mu_mle, sigma_mle))
    
    