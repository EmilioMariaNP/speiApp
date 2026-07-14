r"""
If Kendall's Tau ($\tau$) increases: The drought variables are becoming more "synchronized"—meaning when one is extreme, the other is more likely to be extreme too.
If Upper Tail Dependence ($\lambda_U$) increases: This is the critical "climate signal."
It means that the probability of a "worst-case" intensity occurring given a "worst-case" duration has increased.
"""

import os
import warnings

import numpy as np
from loguru import logger
from utils.utils import load_config, load_dataframe, save_dataframe
import pandas as pd
from scipy import stats
from copulae import (
    GumbelCopula as OriginalGumbelCopula,
    FrankCopula,
    ClaytonCopula,
    StudentCopula,
    GumbelCopula,
    # JoeCopula,
    NormalCopula
)
from fitter import Fitter
import scipy.stats as st
import scipy.optimize as optimize
from copulae import pseudo_obs


# --- HOTFIX FOR NUMPY 2.0+ & COPULAE ---
def patch_copula_method(CopulaClass, method_name):
    if hasattr(CopulaClass, method_name):
        original_method = getattr(CopulaClass, method_name)

        def wrapped_method(self, arg):
            if isinstance(arg, np.ndarray) and arg.size == 1:
                arg = arg.item()
            return original_method(self, arg)

        setattr(CopulaClass, method_name, wrapped_method)


def patch_copula_params(CopulaClass):
    original_prop = getattr(CopulaClass, "params", None)
    if not original_prop:
        return

    def get_params(self):
        return original_prop.fget(self)

    def set_params(self, value):
        if isinstance(value, np.ndarray) and value.size == 1:
            value = value.item()
        original_prop.fset(self, value)

    CopulaClass.params = property(get_params, set_params)


for c in [OriginalGumbelCopula, ClaytonCopula, FrankCopula, GumbelCopula, StudentCopula, NormalCopula]:
    patch_copula_params(c)
    patch_copula_method(c, "itau")

# Patch copulae's squeeze_output for numpy 2.0
import copulae.utility.annotations.reshape as cop_reshape

original_squeeze = cop_reshape.squeeze_output


def patched_squeeze_output(method):
    wrapper = original_squeeze(method)

    def new_wrapper(*args, **kwargs):
        res = wrapper(*args, **kwargs)
        if isinstance(res, np.ndarray) and res.size == 1:
            return float(res.item())
        return res

    return new_wrapper


cop_reshape.squeeze_output = patched_squeeze_output


# also patch it inside the already decorated methods if necessary, or just rely on catching the error locally:
# Actually, the error happens inside `squeeze_output` before we can intercept it!
# So we need to overwrite the `squeeze_output` function in `copulae.utility.annotations.reshape`.
# Wait, the decorator is already applied to `cdf`. We can't un-decorate it easily.
# But we can patch float! We can't patch built-in float.
# Let's intercept `cdf` and `fit` directly on the copula objects.

def patch_copula_cdf(CopulaClass):
    if hasattr(CopulaClass, "cdf"):
        orig_cdf = getattr(CopulaClass, "cdf")

        def safe_cdf(self, x):
            try:
                res = orig_cdf(self, x)
                return res
            except TypeError as e:
                # bypass squeeze_output if it fails
                if "0-dimensional arrays can be converted" in str(e):
                    # call the internal un-decorated cdf if possible, or just
                    res = self._cdf(np.asarray(x)) if hasattr(self, "_cdf") else self.psi(
                        self.ipsi(np.asarray(x)).sum(1))
                    if isinstance(res, np.ndarray) and res.size == 1:
                        return float(res.item())
                    return res
                raise

        setattr(CopulaClass, "cdf", safe_cdf)


for c in [OriginalGumbelCopula, ClaytonCopula, FrankCopula, GumbelCopula, StudentCopula, NormalCopula]:
    patch_copula_cdf(c)


# ---------------------------------------


def select_best_copula(df, columns=None):
    """
    Fits multiple copula families and selects the one with the lowest BIC.
    """
    # 1. Transform data to pseudo-observations (Uniform [0, 1])
    if columns is None:
        columns = ["duration", "intensity"]
    u = pseudo_obs(df[columns])

    # 2. Define candidate families
    # dim=2 for bivariate (Duration, Intensity)
    candidates = [
        GumbelCopula(dim=2),
        FrankCopula(dim=2),
        ClaytonCopula(dim=2),
        # JoeCopula(dim=2),
        NormalCopula(dim=2),
        StudentCopula(dim=2)
    ]

    results = []

    for cop in candidates:
        try:
            # Fit the copula
            cop.fit(u)

            # Record the score (BIC is usually preferred for model selection)
            results.append({
                'family': type(cop).__name__,
                'obj': cop,
                'bic': cop.bic(u),
                'aic': cop.aic(u)
            })
        except Exception as e:
            logger.error(f"Could not fit {type(cop).__name__}: {e}")

    # 3. Sort by BIC and pick the best
    results_df = pd.DataFrame(results).sort_values('bic')
    best_fit = results_df.iloc[0]

    logger.info(f"Best Copula Family: {best_fit['family']} (BIC: {best_fit['bic']:.2f})")

    # best_copula_obj, results_df, best_name
    return best_fit['obj'], results_df, best_fit['family']


def select_best_drought_distribution(df, column_name):
    """
    Fits non-stationary marginal distributions where parameters evolve with time.
    Returns the best fit among: Generalised Extreme Value (GEV), Generalised Pareto Distribution (GPD),
    Time Varying Gamma Distribution, log normal.
    """
    data = df[column_name].dropna().values
    years = df.loc[df[column_name].notna(), 'start_year'].values

    if len(data) == 0:
        return "norm", {"loc": 0, "scale": 1}, None, -1

    t_raw = years - years.min()  # Time covariate starting from 0
    t = t_raw / (t_raw.max() if t_raw.max() > 0 else 1.0)  # Normalize to [0, 1] for optimization stability

    if column_name == "intensity":
        data = np.abs(data)

    results = {}
    dist_filter = globals().get('default_distribution', None)

    # 1. Non-stationary GEV (genextreme) - loc varies with time
    if dist_filter is None or dist_filter == 'genextreme':
        def nll_gev(params):
            c, loc0, loc1, scale = params
            loc = loc0 + loc1 * t
            ll = np.sum(stats.genextreme.logpdf(data, c, loc=loc, scale=scale))
            return -ll if np.isfinite(ll) else 1e10

        try:
            c_init, loc_init, scale_init = stats.genextreme.fit(data)
            c_init = np.clip(c_init, -0.5, 0.5)
        except Exception:
            c_init, loc_init, scale_init = -0.1, np.mean(data), np.std(data)

        bounds_gev = [(-0.5, 0.5), (None, None), (None, None), (0.001, None)]
        res_gev = optimize.minimize(nll_gev, [c_init, loc_init, 0.0, max(scale_init, 0.001)], method='L-BFGS-B',
                                    bounds=bounds_gev)
        if res_gev.success:
            bic = 2 * res_gev.fun + 4 * np.log(len(data))
            results['genextreme'] = {'bic': bic, 'params': res_gev.x}

    # 2. Non-stationary GPD (genpareto) - scale varies with time
    if dist_filter is None or dist_filter == 'genpareto':
        def nll_gpd(params):
            c, loc, scale0, scale1_factor = params
            scale = scale0 * (1 + scale1_factor * t)
            ll = np.sum(stats.genpareto.logpdf(data, c, loc=loc, scale=scale))
            return -ll if np.isfinite(ll) else 1e10

        try:
            c_init, loc_init, scale_init = stats.genpareto.fit(data)
            c_init = np.clip(c_init, -0.5, 0.5)
        except Exception:
            c_init, loc_init, scale_init = 0.1, 0.0, np.std(data)

        bounds_gpd = [(-0.5, 0.5), (None, None), (0.001, None), (-0.99, 10.0)]
        res_gpd = optimize.minimize(nll_gpd, [c_init, loc_init, max(scale_init, 0.001), 0.0], method='L-BFGS-B',
                                    bounds=bounds_gpd)
        if res_gpd.success:
            bic = 2 * res_gpd.fun + 4 * np.log(len(data))
            results['genpareto'] = {'bic': bic, 'params': res_gpd.x}

    # 3. Non-stationary Gamma (gamma) - scale varies with time
    if dist_filter is None or dist_filter == 'gamma':
        def nll_gamma(params):
            a, loc, scale0, scale1_factor = params
            scale = scale0 * (1 + scale1_factor * t)
            ll = np.sum(stats.gamma.logpdf(data, a, loc=loc, scale=scale))
            return -ll if np.isfinite(ll) else 1e10

        try:
            a_init, loc_init, scale_init = stats.gamma.fit(data, floc=0)
            a_init = np.clip(a_init, 0.01, 50)
        except Exception:
            a_init, loc_init, scale_init = 1.0, 0.0, np.mean(data)

        bounds_gamma = [(0.01, 50), (None, None), (0.001, None), (-0.99, 10.0)]
        res_gamma = optimize.minimize(nll_gamma, [a_init, loc_init, max(scale_init, 0.001), 0.0], method='L-BFGS-B',
                                      bounds=bounds_gamma)
        if res_gamma.success:
            bic = 2 * res_gamma.fun + 4 * np.log(len(data))
            results['gamma'] = {'bic': bic, 'params': res_gamma.x}

    # 4. Non-stationary Lognormal (lognorm) - scale varies with time
    if dist_filter is None or dist_filter == 'lognorm':
        def nll_lognorm(params):
            s, loc, scale0, scale1_factor = params
            scale = scale0 * (1 + scale1_factor * t)
            ll = np.sum(stats.lognorm.logpdf(data, s, loc=loc, scale=scale))
            return -ll if np.isfinite(ll) else 1e10

        try:
            s_init, loc_init, scale_init = stats.lognorm.fit(data, floc=0)
            s_init = np.clip(s_init, 0.01, 3.0)
        except Exception:
            s_init, loc_init, scale_init = 1.0, 0.0, np.mean(data)

        bounds_lognorm = [(0.01, 3.0), (None, None), (0.001, None), (-0.99, 10.0)]
        res_lognorm = optimize.minimize(nll_lognorm, [s_init, loc_init, max(scale_init, 0.001), 0.0], method='L-BFGS-B',
                                        bounds=bounds_lognorm)
        if res_lognorm.success:
            bic = 2 * res_lognorm.fun + 4 * np.log(len(data))
            results['lognorm'] = {'bic': bic, 'params': res_lognorm.x}

    # If all fail, fallback to a standard fit
    if not results:
        logger.warning(f"All non-stationary fits failed for {column_name}. Falling back to stationary gamma.")
        try:
            a_init, loc_init, scale_init = stats.gamma.fit(data, floc=0)
        except Exception:
            a_init, loc_init, scale_init = 1.0, 0.0, np.mean(data)
        best_dist_name = 'gamma'
        best_params = {'a': a_init, 'loc': loc_init, 'scale': np.full_like(t, scale_init, dtype=float)}
        return best_dist_name, best_params, None, 0

    # Pick the best distribution
    best_dist_name = min(results, key=lambda k: results[k]['bic'])
    best_opt_params = results[best_dist_name]['params']

    # Map the scalar params to arrays using the normalized t
    if best_dist_name == 'genextreme':
        c, loc0, loc1, scale = best_opt_params
        loc_array = loc0 + loc1 * t
        best_params = {'c': c, 'loc': loc_array, 'scale': scale}
    elif best_dist_name == 'genpareto':
        c, loc, scale0, scale1_factor = best_opt_params
        scale_array = scale0 * (1 + scale1_factor * t)
        best_params = {'c': c, 'loc': loc, 'scale': scale_array}
    elif best_dist_name == 'gamma':
        a, loc, scale0, scale1_factor = best_opt_params
        scale_array = scale0 * (1 + scale1_factor * t)
        best_params = {'a': a, 'loc': loc, 'scale': scale_array}
    elif best_dist_name == 'lognorm':
        s, loc, scale0, scale1_factor = best_opt_params
        scale_array = scale0 * (1 + scale1_factor * t)
        best_params = {'s': s, 'loc': loc, 'scale': scale_array}

    fit_score = 1
    logger.info(
        f"Best Non-Stationary Distribution for {column_name}: {best_dist_name} (BIC: {results[best_dist_name]['bic']:.2f})")

    # 3. Manual Anderson-Darling Check (Optional but Recommended)
    # This helps confirm if the 'best' distribution actually fits the tails well
    fit_score = 0
    try:
        # Get the scipy distribution object
        dist_obj = getattr(st, best_dist_name)
        # For non-stationary distributions, we transform the data to uniform(0,1)
        # using the time-varying parameters (Probability Integral Transform)
        u_t = dist_obj.cdf(data, **best_params)
        # Ensure values are within (0, 1) bounds slightly to avoid numerical issues in AD test
        u_t = np.clip(u_t, 1e-6, 1 - 1e-6)

        logger.info('Running Anderson-Darling test on Uniform-transformed data...')
        res = st.goodness_of_fit(st.uniform,
                                 u_t,
                                 known_params={'loc': 0, 'scale': 1},
                                 n_mc_samples=500,
                                 statistic='ad')
        logger.info(f"GoF Statistic for {best_dist_name} (PIT against Uniform): {res.statistic:.4f}")

        # Interpretation logic
        is_satisfactory = res.pvalue > 0.05
        if is_satisfactory:
            logger.info(f"Fit is statistically significant (p={res.pvalue:.4f})")
            fit_score = 1
        else:
            # Check if it's "close enough" for climate work
            if res.pvalue > 0.01:
                logger.warning(f"Weak fit (p={res.pvalue:.4f}), but likely acceptable for large GCM sets.")
            else:
                logger.warning(f"Poor fit (p={res.pvalue:.4f}). Return periods may be unreliable.")
                fit_score = -1

    except Exception as e:
        logger.warning(f"Could not calculate specific GoF: {e}")

    return best_dist_name, best_params, None, fit_score


def run_drought_copula_analysis(events_df, cutoff_year, copula_family="gumbel"):
    events_df = events_df.copy()
    events_df["intensity_abs"] = events_df["intensity"].abs()

    ref = events_df[events_df["start_year"] < cutoff_year]
    proj = events_df[events_df["start_year"] >= cutoff_year]

    periods = {"Reference": ref, "Projection": proj}
    results = {}

    for name, data in periods.items():
        if len(data) < 10:  # Increased minimum sample size for stability
            logger.warning(f"Warning: Not enough data for {name} period.")
            continue

        # 1. Marginal Fitting and Transformation
        u_list = []
        for col in ["duration", "intensity_abs"]:
            params = stats.gamma.fit(data[col], floc=0)
            u_vars = stats.gamma.cdf(data[col], *params)
            u_list.append(u_vars)

        # 2. Force U into a clean NumPy array of floats
        U = np.column_stack(u_list).astype(np.float64)

        try:
            # 3. Initialize and Fit
            # dim=2 for bivariate (Duration, Intensity)
            family = copula_family.lower()
            if family == "gumbel":
                cop = OriginalGumbelCopula(dim=2)
            elif family == "clayton":
                cop = ClaytonCopula(dim=2)
            elif family == "frank":
                cop = FrankCopula(dim=2)
            elif family in ["student-t", "t"]:
                cop = StudentCopula(dim=2)
            elif family == "normal":
                cop = NormalCopula(dim=2)
            else:
                raise ValueError(f"Unknown copula family: {copula_family}")

            # Use 'ml' (Maximum Likelihood) method explicitly
            # If it still fails, try method='mpl'
            cop.fit(U, method="ml")

            # 4. Extract Metrics
            # Note: cop.tau and cop.lambda_ return values based on the fitted parameter
            results[name] = {
                "Copula Family": copula_family.capitalize(),
                "Parameter": float(cop.params),
                "Kendall_Tau": float(cop.tau),
                "Upper_Tail_Dep": float(cop.lambda_[1]),
            }
        except Exception as e:
            logger.error(f"Error fitting {name} period: {e}")

    return pd.DataFrame(results).T


def analyze_return_periods(events_df,
                           ref_start_year,
                           ref_end_year,
                           proj_end_year,
                           proj_start_year,
                           return_periods,
                           copula_family=None):
    # 1. Prepare Data
    events_df = events_df.copy()
    # events_df["intensity_abs"] = events_df["intensity"].abs()

    ref_data = events_df[(events_df["start_year"] >= ref_start_year) & (events_df["start_year"] <= ref_end_year)]
    proj_data = events_df[(events_df["start_year"] >= proj_start_year) & (events_df["start_year"] <= proj_end_year)]

    # Calculate average events per year (E) for each period
    ref_years = ref_data["start_year"].max() - ref_data["start_year"].min() + 1
    proj_years = proj_data["start_year"].max() - proj_data["start_year"].min() + 1

    e_ref = len(ref_data) / ref_years
    e_proj = len(proj_data) / proj_years

    variables = ["duration", "intensity"]

    # 2. Fit Marginals
    dist_info = {}
    fit_scores_dict = {}
    for name, data in [("Ref", ref_data), ("Proj", proj_data)]:
        dist_info[name] = {}
        for col in variables:
            logger.info(f'Fitting marginal distribution for {name} period, {col}..')
            best_dist_name, best_params, _, fit_score = select_best_drought_distribution(
                data, col
            )  # automatically select the best marginal distribution
            dist_info[name][col] = {"name": best_dist_name, "params": best_params}
            fit_scores_dict[f'{name}_{col}_fit_score'] = fit_score
            logger.info('___________________________________________________________________\n')

    def apply_dist(dist_dict, method, val):
        dist = getattr(stats, dist_dict["name"])
        func = getattr(dist, method)

        if not np.isscalar(val):
            val = np.asarray(val)

        if isinstance(dist_dict["params"], dict):
            params = dist_dict["params"]
            if np.isscalar(val):
                # For scalars (thresholds), use the last year's parameters
                params = {k: (v[-1] if isinstance(v, np.ndarray) else v) for k, v in params.items()}
            return func(val, **params)
        return func(val, *dist_dict["params"])

    # 3. Fit Copulas

    def get_copula(data, dist_info_subset):
        u = np.column_stack(
            [apply_dist(dist_info_subset[col], "cdf", data[col]) for col in variables]
        )
        # Ensure u is within (0, 1) to avoid numerical issues in fitting
        u = np.clip(u, 0.00001, 0.99999)

        cop = None
        if copula_family is None:
            # Find the best copula family for the data
            logger.info('Searching best copula family...')
            cop, _cop_rank_df, copula_family_selected = select_best_copula(u)

        else:
            copula_family_selected = copula_family
            _cop_rank_df = None
            logger.info(f'Fitting copula using {copula_family_selected} family...')
            try:
                # 3. Initialize and Fit
                # dim=2 for bivariate (Duration, Intensity)
                family = copula_family.lower()
                if family == "gumbel":
                    cop = OriginalGumbelCopula(dim=2)
                elif family == "clayton":
                    cop = ClaytonCopula(dim=2)
                elif family == "frank":
                    cop = FrankCopula(dim=2)
                elif family in ["student-t", "t"]:
                    cop = StudentCopula(dim=2)
                elif family == "normal":
                    cop = NormalCopula(dim=2)
                else:
                    raise ValueError(f"Unknown copula family: {copula_family}")

                # Use 'ml' (Maximum Likelihood) method explicitly
                cop.fit(u, method="ml")
            except Exception as e:
                logger.warning(f"Error fitting {name} period with ML ({e}). Falling back to 'itau' method.")
                try:
                    cop.fit(u, method="itau")
                except Exception as e2:
                    logger.error(f"Error fitting {name} period with itau: {e2}")

        return cop, _cop_rank_df, copula_family_selected

    cop_ref, _cop_rank_df, copula_family_selected_ref = get_copula(
        ref_data, dist_info["Ref"]
    )
    cop_proj, _cop_rank_df, copula_family_selected_proj = get_copula(
        proj_data, dist_info["Proj"]
    )

    comparison_results = []

    for T in return_periods:
        # Probability of exceedance per event: p = 1 / (T * E)
        p_exceed = 1 / (T * e_ref)
        p_non_exceed = 1 - p_exceed

        # Ensure probability is within [0, 1]
        p_non_exceed = max(0.001, min(0.999, p_non_exceed))

        # Probability of exceedance for projection
        p_exceed_proj = 1 / (T * e_proj)
        p_non_exceed_proj = max(0.001, min(0.999, 1 - p_exceed_proj))

        # Thresholds from Historical Period
        d_thresh = apply_dist(dist_info["Ref"]["duration"], "ppf", p_non_exceed)
        i_thresh = apply_dist(dist_info["Ref"]["intensity"], "ppf", p_non_exceed)

        # Thresholds from Projection Period
        d_thresh_proj = apply_dist(
            dist_info["Proj"]["duration"], "ppf", p_non_exceed_proj
        )
        i_thresh_proj = apply_dist(
            dist_info["Proj"]["intensity"], "ppf", p_non_exceed_proj
        )

        # Joint Probability in Reference (using survival copula formula)
        # P(D > d and I > i) = 1 - u - v + C(u, v)
        u_ref = float(apply_dist(dist_info["Ref"]["duration"], "cdf", d_thresh))
        v_ref = float(apply_dist(dist_info["Ref"]["intensity"], "cdf", i_thresh))

        # Hack to bypass numpy 2.0 TypeError in copulae's squeeze_output: pass array of length 2
        c_ref_arr = cop_ref.cdf([[u_ref, v_ref], [u_ref, v_ref]])
        c_ref_val = float(c_ref_arr[0])

        p_joint_ref = 1 - u_ref - v_ref + c_ref_val
        p_joint_ref = max(p_joint_ref, 1e-8)  # prevent division by zero
        t_joint_ref = min(1 / (p_joint_ref * e_ref), 10000.0)  # cap at 10,000 years

        # Joint Probability in Projection for the SAME thresholds
        u_proj = float(apply_dist(dist_info["Proj"]["duration"], "cdf", d_thresh))
        v_proj = float(apply_dist(dist_info["Proj"]["intensity"], "cdf", i_thresh))

        c_proj_arr = cop_proj.cdf([[u_proj, v_proj], [u_proj, v_proj]])
        c_proj_val = float(c_proj_arr[0])

        p_joint_proj = 1 - u_proj - v_proj + c_proj_val
        p_joint_proj = max(p_joint_proj, 1e-8)  # prevent division by zero
        t_joint_proj = min(1 / (p_joint_proj * e_proj), 10000.0)  # cap at 10,000 years

        comparison_results.append(
            {
                "Return period": T,
                "Hist_Duration_Thresh": round(d_thresh, 1),
                "Hist_Intensity_Thresh": round(i_thresh, 1),
                "Proj_Duration_Thresh": round(d_thresh_proj, 1),
                "Proj_Intensity_Thresh": round(i_thresh_proj, 1),
                "Ref_Joint_T": round(t_joint_ref, 1),
                "Proj_Joint_T": round(t_joint_proj, 1),
                "Likelihood_Change": round(t_joint_ref / t_joint_proj, 2),
                "Copula_Family_Ref": copula_family_selected_ref,
                "Copula_Family_Proj": copula_family_selected_proj,
                'dist_info_dur_ref': dist_info["Ref"]["duration"]['name'],
                'dist_info_intensity_ref': dist_info["Ref"]["intensity"]['name'],
                'dist_info_dur_proj': dist_info["Proj"]["duration"]['name'],
                'dist_info_intensity_proj': dist_info["Proj"]["intensity"]['name']
            }
        )
        for k_col, v_fit in fit_scores_dict.items():
            comparison_results[-1][k_col] = v_fit

    return pd.DataFrame(comparison_results)


def analyze_univariate_return_periods(df,
                                      column_name,
                                      spei_col,
                                      ref_start_year,
                                      ref_end_year,
                                      proj_end_year,
                                      proj_start_year,
                                      growth_season_months,
                                      return_periods
                                      ):
    """
    Calculates return periods for a specific drought variable and
    compares historical vs. future frequency.
    """
    # 1. Prepare Data
    data = df.sort_values(by='start_year').copy()

    # Apply growth season filter for SPEI-3 to match joint analysis logic
    if spei_col == "spei3":
        data = data[(data['start_month'].isin(growth_season_months)) |
                    (data['end_month'].isin(growth_season_months))

                    ]

    if column_name == "intensity":
        data[column_name] = data[column_name].abs()

    ref_df = data[(data["start_year"] >= ref_start_year) & (data["start_year"] <= ref_end_year)]
    proj_df = data[(data["start_year"] >= proj_start_year) & (data["start_year"] <= proj_end_year)]

    # Calculate average events per year (lambda)
    def get_lambda(df_subset):
        years = df_subset["start_year"].max() - df_subset["start_year"].min() + 1
        return len(df_subset) / years

    lambda_ref = get_lambda(ref_df)
    lambda_proj = get_lambda(proj_df)

    # 2. Fit Marginal Distributions
    logger.info(f'Fitting marginal distribution for reference period {column_name}...')
    best_dist_name_ref, best_params_ref, _, ref_fit_score = select_best_drought_distribution(
        ref_df, column_name
    )

    logger.info(f'Fitting marginal distribution for projected {column_name}...')
    best_dist_name_proj, best_params_proj, _, proj_fit_score = select_best_drought_distribution(
        proj_df, column_name
    )

    dist_ref = getattr(stats, best_dist_name_ref)
    dist_proj = getattr(stats, best_dist_name_proj)

    results = []

    for T in return_periods:
        # 3. Determine the historical threshold for T years
        # p = 1 - 1/(lambda * T)
        p_ref = 1 - (1 / (lambda_ref * T))

        # Ensure p is valid for PPF
        if p_ref < 0:
            p_ref = 0.01

        # Calculate the threshold value (x) from historical distribution
        if isinstance(best_params_ref, dict):
            scalar_params_ref = {k: (v[-1] if isinstance(v, np.ndarray) else v) for k, v in best_params_ref.items()}
            threshold = dist_ref.ppf(p_ref, **scalar_params_ref)
        else:
            threshold = dist_ref.ppf(p_ref, *best_params_ref)

        # 4. Calculate the 'new' return period for that same threshold in the future
        # Find p_proj = F(threshold | future_params)
        if isinstance(best_params_proj, dict):
            scalar_params_proj = {k: (v[-1] if isinstance(v, np.ndarray) else v) for k, v in best_params_proj.items()}
            p_proj = dist_proj.cdf(threshold, **scalar_params_proj)
        else:
            p_proj = dist_proj.cdf(threshold, *best_params_proj)

        # T_proj = 1 / (lambda_proj * (1 - p_proj))
        # Handle edge case where p_proj is 1.0
        exceedance_prob_proj = 1 - p_proj
        if exceedance_prob_proj > 0:
            t_proj = 1 / (lambda_proj * exceedance_prob_proj)
        else:
            t_proj = 0  # np.inf

        # 5. Likelihood Change
        # Ratio > 1 means the event is more frequent (Risk is higher)
        likelihood_ratio = T / t_proj if t_proj != 0 else 0  # np.inf

        results.append(
            {
                "Return Period (Years)": T,
                f"{column_name} Historical Threshold": round(threshold, 2),
                f"{column_name} Future Return Period": round(t_proj, 2),
                f"{column_name} Likelihood Change (x)": round(likelihood_ratio, 2),
                f'{column_name}_ref_fit_score': ref_fit_score,  # -1 poor/unreliable, 0 acceptable, 1 good
                f'{column_name}_proj_fit_score': proj_fit_score
            }
        )

    return pd.DataFrame(results)


def main_copula_nn_station(config):

    ref_start_year = config["ref_start_year"]
    ref_end_year = config["ref_end_year"]
    proj_end_year = config["proj_end_year"]  # analysis will be limited to the data before this year
    proj_start_year = config['proj_start_year']

    spatial_unit_col = config["spatial_unit_col"]

    # datasets_data = config["datasets"]
    # datasets_keys = list(datasets_data.keys())
    # validation_datasets = config["validation_datasets"]
    # proj_datasets_keys = [x for x in datasets_keys if x not in validation_datasets]

    # ref_scenario = config['datasets']['projections']['reference_scenario']
    # analysis_scenarios = config['scenario']
    # if len(analysis_scenarios) > 0 and ref_scenario not in analysis_scenarios:
    #     analysis_scenarios.append(ref_scenario)

    spei_scales = config.get('spei_indices', [3, 12])

    copula_family = config.get('copula_family', 'gumbel')
    default_distribution = config.get('default_distribution', None)  # Set to None to use dynamic distribution selection
    analysis_regions = config.get('spatial_units', [])
    exclude_month_events = config.get('exclude_month_events', True)  # exclude drought events of 1 month length --> create noise and are not necessarily drought events

    # use the whole year if growth season not provided
    growth_season_start = config.get('growth_season_start', 1)
    growth_season_end = config.get('growth_season_end', 12)
    growth_season_months = list(range(growth_season_start, growth_season_end + 1))

    return_periods = config.get('return_periods', [5, 10])
    ref_scenario = config.get('reference_scenario', 'historical')
    analysis_scenarios = config.get('analysis_scenarios', [])

    events_csv = config["dry_events_csv"]
    copula_csv = config.get('copula_analysis_csv', None)
    gcm_eval_csv = config.get('gcm_eval_csv', None)
    return_periods_csv = config["return_periods_csv"]
    # plots_dir = os.path.join(work_dir, 'frequency', 'plots')


    logger.info('Starting non-stationary copula analysis...')


    df_events = load_dataframe(events_csv)
    gcm_eval_df = load_dataframe(gcm_eval_csv)

    print()

    # remove validation
    df_events = df_events[df_events['scenario'] != 'validation']

    if exclude_month_events:
        df_events = df_events[df_events['duration'] > 1].copy(deep=True)
    df_events.sort_values(by=["start_year"], inplace=True)
    df_events = df_events[(df_events["start_year"] <= proj_end_year) & (df_events['start_year'] >= ref_start_year)]

    gcms = df_events["gcm"].unique().tolist()

    if len(analysis_regions) == 0:
        spatial_units = df_events[spatial_unit_col].unique().tolist()
    else:
        spatial_units = analysis_regions  # limits the analysis to a subset of regions

    if len(analysis_scenarios) == 0:
        scenarios = df_events["scenario"].unique().tolist()
        scenarios.remove(ref_scenario)
    else:
        scenarios = analysis_scenarios

    df_copuls_list = []
    # df_rp_list = []
    for su in spatial_units:
        for spei_scale in spei_scales:
            spei_col = f"spei{spei_scale}"
            for scenario in scenarios:
                for gcm in gcms:
                    logger.info(f"Evaluating {scenario}-{gcm}-{spei_col} for {su}")

                    # read from the GCM evaluation file if the GCM is fit to model the su
                    gcm_eval = gcm_eval_df[
                                    (gcm_eval_df["gcm"] == gcm) &
                                    (gcm_eval_df[spatial_unit_col] == su) &
                                    (gcm_eval_df['spei'] == spei_col)
                        ]

                    if len(gcm_eval.index) >0:
                        gcm_is_valid = gcm_eval.iloc[0]['passed']
                    else:
                        gcm_is_valid = True

                    if gcm_is_valid:

                        df_cop = df_events[
                            (df_events["scenario"].isin([scenario, ref_scenario]) )
                            & (df_events["gcm"] == gcm)
                            & (df_events["spei"] == spei_scale)
                            & (df_events[spatial_unit_col] == su)

                            ]

                        if spei_scale == 3:  # limit spei 3 to growth season months only
                            df_cop = df_cop[(df_cop['start_month'].isin(growth_season_months)) |
                                            (df_cop['end_month'].isin(growth_season_months))
                                            ]

                        logger.info("Analysing join distribution...")
                        df_cop_analysis = analyze_return_periods(
                            events_df=df_cop,
                            copula_family=copula_family,
                            ref_start_year=ref_start_year,
                            ref_end_year=ref_end_year,
                            proj_end_year=proj_end_year,
                            proj_start_year=proj_start_year,
                            return_periods=return_periods
                        )

                        df_cop_analysis["scenario"] = scenario
                        df_cop_analysis["gcm"] = gcm
                        df_cop_analysis["spei"] = spei_scale
                        df_cop_analysis[spatial_unit_col] = su
                        df_copuls_list.append(df_cop_analysis)
                    else:
                        logger.info(f"GCM {gcm} not fit for {spei_col} in {su}. Skipping the GCM.")


    df_copulas = pd.concat(df_copuls_list, ignore_index=True)
    save_dataframe(df=df_copulas, csv_file=copula_csv)


if __name__ == "__main__":
    # config_file = "../data/ecoregions/workflow_config_ecoregions.json"
    config_file = "../data/config_ecoregions.json"

    config = load_config(config_file)


    home_dir = config.get("home_dir", None)
    if home_dir:

        config["dry_events_csv"] = os.path.join(
            home_dir, config["dry_events_csv"]
        )

        config['gcm_eval_csv'] = os.path.join(home_dir, config['gcm_eval_csv'])
        config['copula_analysis_csv'] = os.path.join(home_dir, config['copula_analysis_csv'])


    main_copula_nn_station(config)

    print("Process completed.")