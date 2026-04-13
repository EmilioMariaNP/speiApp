import numpy as np
from scipy.optimize import minimize
from scipy.stats import gamma, expon, norm, kstest
import warnings

# Suppress some specific warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

class Copula:
    def __init__(self, theta=None):
        self.theta = theta
        self.aic = np.inf
        self.bic = np.inf
        self.name = "BaseCopula"

    def fit(self, u, v):
        """Fit theta using Maximum Likelihood Estimation."""
        # Initial guess
        initial_theta = self._get_initial_theta()
        
        # Bounds
        bounds = self._get_bounds()
        
        def neg_log_likelihood(theta):
            # Penalize invalid theta
            if not self._is_valid_theta(theta[0]):
                return 1e10
            
            # Calculate PDF
            pdf_vals = self._pdf(u, v, theta[0])
            
            # Avoid log(0)
            pdf_vals = np.maximum(pdf_vals, 1e-10)
            return -np.sum(np.log(pdf_vals))

        try:
            result = minimize(neg_log_likelihood, [initial_theta], bounds=bounds, method='L-BFGS-B')
            if result.success:
                self.theta = result.x[0]
                k = 1 # number of parameters
                n = len(u)
                log_L = -result.fun
                self.aic = 2 * k - 2 * log_L
                self.bic = k * np.log(n) - 2 * log_L
            else:
                self.theta = initial_theta
                # print(f"Optimization failed for {self.name}: {result.message}")
        except Exception as e:
            # print(f"Error fitting {self.name}: {e}")
            self.theta = initial_theta

    def cdf(self, u, v):
        if self.theta is None:
            raise ValueError("Copula not fitted")
        return self._cdf(u, v, self.theta)

    def _get_initial_theta(self):
        return 0.5
        
    def _is_valid_theta(self, theta):
        return True

    def _get_bounds(self):
        return [(None, None)]

    def _cdf(self, u, v, theta):
        raise NotImplementedError

    def _pdf(self, u, v, theta):
        raise NotImplementedError

class ClaytonCopula(Copula):
    def __init__(self):
        super().__init__()
        self.name = "Clayton"

    def _get_initial_theta(self):
        return 2.0

    def _get_bounds(self):
        return [(0.01, 20.0)] # Theta > 0

    def _is_valid_theta(self, theta):
        return theta > 0

    def _cdf(self, u, v, theta):
        # C(u,v) = (u^-theta + v^-theta - 1)^(-1/theta)
        return np.maximum(u**(-theta) + v**(-theta) - 1, 0)**(-1/theta)

    def _pdf(self, u, v, theta):
        # c(u,v) = (1+theta) * (u*v)^(-1-theta) * (u^-theta + v^-theta - 1)^(-1/theta - 2)
        term1 = 1 + theta
        term2 = (u * v)**(-1 - theta)
        term3 = np.maximum(u**(-theta) + v**(-theta) - 1, 1e-10)**(-1/theta - 2)
        return term1 * term2 * term3

class GumbelCopula(Copula):
    def __init__(self):
        super().__init__()
        self.name = "Gumbel"

    def _get_initial_theta(self):
        return 2.0

    def _get_bounds(self):
        return [(1.01, 20.0)] # Theta >= 1

    def _is_valid_theta(self, theta):
        return theta >= 1

    def _cdf(self, u, v, theta):
        # C(u,v) = exp(-((-ln u)^theta + (-ln v)^theta)^(1/theta))
        neg_ln_u = -np.log(u)
        neg_ln_v = -np.log(v)
        return np.exp(-(neg_ln_u**theta + neg_ln_v**theta)**(1/theta))

    def _pdf(self, u, v, theta):
        # Complex derivative... check implementation or approx
        # Using numerical gradient might be safer but slow.
        # Analytical:
        x = -np.log(u)
        y = -np.log(v)
        
        tmp = x**theta + y**theta
        tmp_pow = tmp**(1/theta)
        
        C = np.exp(-tmp_pow)
        
        term1 = C * (u * v)**(-1)
        term2 = tmp**(2/theta - 2)
        term3 = (x * y)**(theta - 1)
        term4 = 1 + (theta - 1) * tmp**(-1/theta)
        
        return term1 * term2 * term3 * term4

class FrankCopula(Copula):
    def __init__(self):
        super().__init__()
        self.name = "Frank"

    def _get_initial_theta(self):
        return 5.0

    def _get_bounds(self):
        return [(-50.0, 50.0)] # Theta != 0

    def _is_valid_theta(self, theta):
        return abs(theta) > 1e-6

    def _cdf(self, u, v, theta):
        # C(u,v) = -1/theta * ln(1 + (exp(-theta u)-1)(exp(-theta v)-1)/(exp(-theta)-1))
        num = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
        den = np.exp(-theta) - 1
        return -1/theta * np.log(1 + num/den)

    def _pdf(self, u, v, theta):
        num = -theta * (np.exp(-theta) - 1) * np.exp(-theta * (u + v))
        den_part = (np.exp(-theta) - 1) + (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
        return num / (den_part**2)

class GaussianCopula(Copula):
    # Simplified Gaussian Copula (scalar rho)
    def __init__(self):
        super().__init__()
        self.name = "Gaussian"

    def _get_initial_theta(self):
        return 0.5 

    def _get_bounds(self):
        return [(-0.99, 0.99)]

    def _is_valid_theta(self, theta):
        return -1 < theta < 1

    def _cdf(self, u, v, theta):
        # Not easily analytical, usually calls mvn
        # Skipping CDF for simplicity unless needed for plotting, 
        # but calculate_return_period usually uses CDF.
        # We can implement using scipy.stats.multivariate_normal
        from scipy.stats import multivariate_normal, norm
        x = norm.ppf(u)
        y = norm.ppf(v)
        # This is slow per point. Vectorization needed.
        # But we only need it for return periods on a grid.
        # Placeholder
        return u * v # Fallback

    def _pdf(self, u, v, theta):
        # c(u,v) = 1/sqrt(1-rho^2) * exp( (2*rho*x*y - rho^2*(x^2+y^2)) / (2*(1-rho^2)) )
        # where x, y = norm.ppf(u), norm.ppf(v)
        rho = theta
        x = norm.ppf(u)
        y = norm.ppf(v)
        
        term1 = 1 / np.sqrt(1 - rho**2)
        num = 2 * rho * x * y - rho**2 * (x**2 + y**2)
        den = 2 * (1 - rho**2)
        return term1 * np.exp(num / den)

class IndependenceCopula(Copula):
    def __init__(self):
        super().__init__()
        self.name = "Independence"
        self.theta = 0

    def fit(self, u, v):
        self.aic = 0 # Not strictly 0, but effectively 0 params
        self.bic = 0

    def _cdf(self, u, v, theta):
        return u * v

    def _pdf(self, u, v, theta):
        return np.ones_like(u)

# --- Utilities ---

def fit_marginals(data, dist_names=['gamma', 'expon']):
    """
    Fits marginal distributions and selects best by KS test or AIC.
    Returns (best_dist_name, params, cdf_func)
    """
    best_dist = None
    best_params = None
    best_p_value = -1
    best_obj = None

    # Filter data > 0 for gamma/expon and ensure finite
    data_clean = data[(data > 0) & (np.isfinite(data))]
    if len(data_clean) < 5:
        # Fallback to empirical
        return "empirical", None, None

    for name in dist_names:
        if name == 'gamma':
            params = gamma.fit(data_clean)
            # KS test
            stat, p = kstest(data_clean, 'gamma', args=params)
        elif name == 'expon':
            params = expon.fit(data_clean)
            stat, p = kstest(data_clean, 'expon', args=params)
        
        if p > best_p_value:
            best_p_value = p
            best_dist = name
            best_params = params

    return best_dist, best_params, None

def transform_to_uniform(data, dist_name, params):
    """Transforms real data to u in [0,1] using fitted CDF."""
    if dist_name == 'gamma':
        return gamma.cdf(data, *params)
    elif dist_name == 'expon':
        return expon.cdf(data, *params)
    elif dist_name == 'empirical':
        # Plotting position formula: (rank - 0.44) / (n + 0.12) or similar
        # Weibull: rank / (n+1)
        ranks = data.rank()
        n = len(data)
        return ranks / (n + 1)
    else:
        return data

def get_best_copula(u, v):
    copulas = [
        ClaytonCopula(),
        GumbelCopula(),
        FrankCopula(),
        IndependenceCopula() 
        # GaussianCopula() # Skip Gaussian for now to avoid complexity
    ]
    
    best_cop = None
    min_aic = np.inf
    
    for cop in copulas:
        cop.fit(u, v)
        if cop.aic < min_aic:
            min_aic = cop.aic
            best_cop = cop
            
    return best_cop

def calculate_return_period(copula, u_crit, v_crit, type='AND'):
    """
    T_AND = 1 / (1 - u - v + C(u,v)) * Interval
    T_OR = 1 / (1 - C(u,v)) * Interval
    Interval is mean inter-arrival time (usually handled by the fact we model events).
    RP is in "number of events". To get years, divide by expected events/year.
    """
    # Note: u_crit and v_crit are CDF values (probabilities of non-exceedance)
    # But usually RP is for Exceedance.
    # High Duration (D > d) -> u_crit is close to 1?
    # Yes, F_D(d) = u. So P(D > d) = 1 - u.
    
    C_uv = copula.cdf(u_crit, v_crit)
    
    if type == 'AND':
        # P(U > u and V > v) = 1 - u - v + C(u,v)
        prob = 1 - u_crit - v_crit + C_uv
    else: # OR
        # P(U > u or V > v) = 1 - C(u,v)
        prob = 1 - C_uv
        
    if prob <= 0: return np.inf
    return 1.0 / prob
