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
import pyvinecopulib as pv
from fitter import Fitter
import scipy.stats as st
import scipy.optimize as optimize

def get_upper_tail_dependence(cop, u=0.9999):
    c_val = cop.cdf(np.array([[u, u]]))
    return max(0.0, min((1 - 2*u + c_val[0]) / (1 - u), 1.0))


# ---------------------------------------


def select_best_copula(df, columns=None):
    """
    Fits multiple copula families and selects the one with the lowest BIC using pyvinecopulib.
    """
    if isinstance(df, np.ndarray):
        u = df
    else:
        if columns is None:
            columns = ["duration", "intensity"]
        u = pv.to_pseudo_obs(df[columns].values)

    family_set = [
        pv.BicopFamily.gaussian, pv.BicopFamily.student, pv.BicopFamily.clayton,
        pv.BicopFamily.gumbel, pv.BicopFamily.frank, pv.BicopFamily.joe,
        pv.BicopFamily.bb1, pv.BicopFamily.bb6, pv.BicopFamily.bb7, pv.BicopFamily.bb8,
        pv.BicopFamily.tawn
    ]
    controls = pv.FitControlsBicop(family_set=family_set, selection_criterion="bic")
    
    cop = pv.Bicop()
    cop.select(data=u, controls=controls)

    logger.info(f"Best Copula Family: {cop.family.name} (BIC: {cop.bic(u):.2f})")
    
    # Return the copula object, None for rank_df (no longer needed), and family name
    return cop, None, cop.family.name


def select_best_drought_distribution(df, column_name, is_reference_period=False):
    """
    Fits marginal distributions for drought variables.
    For the reference period (is_reference_period=True), both stationary and non-stationary models compete via BIC.
    For projection periods (is_reference_period=False), strictly non-stationary time-varying models are evaluated.
    Evaluates: GEV, GPD, Gamma, Lognorm, Weibull, Pearson Type III, and Exponential.
    """
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Sanitize data: remove non-finite entries (np.inf, -np.inf, np.nan) and non-positive numerical artifacts
    valid_mask = np.isfinite(df[column_name]) & (df[column_name] > 0) & np.isfinite(df['start_year'])
    data = df.loc[valid_mask, column_name].values
    years = df.loc[valid_mask, 'start_year'].values

    if len(data) == 0:
        return "norm", {"loc": 0, "scale": 1}, None, -1

    t_raw = years - years.min()  # Time covariate starting from 0
    t = t_raw / (t_raw.max() if t_raw.max() > 0 else 1.0)  # Normalize to [0, 1] for optimization stability

    if column_name == "intensity":
        data = np.abs(data)

    results = {}
    param_map = {
        'genextreme': ['c', 'loc', 'scale'],
        'genpareto': ['c', 'loc', 'scale'],
        'gamma': ['a', 'loc', 'scale'],
        'lognorm': ['s', 'loc', 'scale'],
        'weibull_min': ['c', 'loc', 'scale'],
        'pearson3': ['skew', 'loc', 'scale'],
        'expon': ['loc', 'scale'],
        'norm': ['loc', 'scale']
    }

    # --- 1. Stationary fits (Only permitted for Reference Period) ---
    if is_reference_period:
        for d_name in ['genextreme', 'genpareto', 'gamma', 'lognorm', 'weibull_min', 'pearson3', 'expon']:
            try:
                dist_obj = getattr(st, d_name)
                if d_name in ['gamma', 'lognorm', 'weibull_min', 'expon']:
                    params = dist_obj.fit(data, floc=0)
                else:
                    params = dist_obj.fit(data)

                ll = np.sum(dist_obj.logpdf(data, *params))
                if np.isfinite(ll):
                    k = len(params) - (1 if d_name in ['gamma', 'lognorm', 'weibull_min', 'expon'] else 0)
                    bic = -2 * ll + k * np.log(len(data))
                    param_names = param_map[d_name]
                    param_dict = dict(zip(param_names, params))
                    if 'scale' in param_dict and not isinstance(param_dict['scale'], np.ndarray):
                        param_dict['scale'] = np.full_like(t, param_dict['scale'], dtype=float)
                    results[f"{d_name}_stat"] = {'bic': bic, 'params': param_dict, 'dist': d_name, 'type': 'stat'}
            except Exception as e:
                logger.debug(f"Stationary fit failed for {d_name}: {e}")

    # --- 2. Non-Stationary fits ---
    # 2.1 Non-stationary GEV (genextreme) - loc varies with time
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
    res_gev = optimize.minimize(nll_gev, [c_init, loc_init, 0.0, max(scale_init, 0.001)], method='L-BFGS-B', bounds=bounds_gev)
    if res_gev.success:
        bic = 2 * res_gev.fun + 4 * np.log(len(data))
        c, loc0, loc1, scale = res_gev.x
        results['genextreme_nn'] = {'bic': bic, 'params': {'c': c, 'loc': loc0 + loc1 * t, 'scale': scale}, 'dist': 'genextreme', 'type': 'nn'}

    # 2.2 Non-stationary GPD (genpareto) - scale varies with time (loc=0 fixed to prevent threshold singularity)
    def nll_gpd(params):
        c, scale0, scale1_factor = params
        scale = scale0 * (1 + scale1_factor * t)
        ll = np.sum(stats.genpareto.logpdf(data, c, loc=0.0, scale=scale))
        return -ll if np.isfinite(ll) else 1e10

    try:
        c_init, _, scale_init = stats.genpareto.fit(data, floc=0)
        c_init = np.clip(c_init, -0.5, 0.5)
    except Exception:
        c_init, scale_init = 0.1, np.std(data)

    bounds_gpd = [(-0.5, 0.5), (0.001, None), (-0.99, 10.0)]
    res_gpd = optimize.minimize(nll_gpd, [c_init, max(scale_init, 0.001), 0.0], method='L-BFGS-B', bounds=bounds_gpd)
    if res_gpd.success:
        bic = 2 * res_gpd.fun + 3 * np.log(len(data))
        c, scale0, scale1_factor = res_gpd.x
        results['genpareto_nn'] = {'bic': bic, 'params': {'c': c, 'loc': 0.0, 'scale': scale0 * (1 + scale1_factor * t)}, 'dist': 'genpareto', 'type': 'nn'}

    # 2.3 Non-stationary Gamma (gamma) - scale varies with time (loc=0 fixed to prevent threshold singularity)
    def nll_gamma(params):
        a, scale0, scale1_factor = params
        scale = scale0 * (1 + scale1_factor * t)
        ll = np.sum(stats.gamma.logpdf(data, a, loc=0.0, scale=scale))
        return -ll if np.isfinite(ll) else 1e10

    try:
        a_init, _, scale_init = stats.gamma.fit(data, floc=0)
        a_init = np.clip(a_init, 0.01, 50)
    except Exception:
        a_init, scale_init = 1.0, np.mean(data)

    bounds_gamma = [(0.01, 50), (0.001, None), (-0.99, 10.0)]
    res_gamma = optimize.minimize(nll_gamma, [a_init, max(scale_init, 0.001), 0.0], method='L-BFGS-B', bounds=bounds_gamma)
    if res_gamma.success:
        bic = 2 * res_gamma.fun + 3 * np.log(len(data))
        a, scale0, scale1_factor = res_gamma.x
        results['gamma_nn'] = {'bic': bic, 'params': {'a': a, 'loc': 0.0, 'scale': scale0 * (1 + scale1_factor * t)}, 'dist': 'gamma', 'type': 'nn'}

    # 2.4 Non-stationary Lognormal (lognorm) - scale varies with time (loc=0 fixed to prevent threshold singularity)
    def nll_lognorm(params):
        s, scale0, scale1_factor = params
        scale = scale0 * (1 + scale1_factor * t)
        ll = np.sum(stats.lognorm.logpdf(data, s, loc=0.0, scale=scale))
        return -ll if np.isfinite(ll) else 1e10

    try:
        s_init, _, scale_init = stats.lognorm.fit(data, floc=0)
        s_init = np.clip(s_init, 0.01, 3.0)
    except Exception:
        s_init, scale_init = 1.0, np.mean(data)

    bounds_lognorm = [(0.01, 3.0), (0.001, None), (-0.99, 10.0)]
    res_lognorm = optimize.minimize(nll_lognorm, [s_init, max(scale_init, 0.001), 0.0], method='L-BFGS-B', bounds=bounds_lognorm)
    if res_lognorm.success:
        bic = 2 * res_lognorm.fun + 3 * np.log(len(data))
        s, scale0, scale1_factor = res_lognorm.x
        results['lognorm_nn'] = {'bic': bic, 'params': {'s': s, 'loc': 0.0, 'scale': scale0 * (1 + scale1_factor * t)}, 'dist': 'lognorm', 'type': 'nn'}

    # 2.5 Non-stationary Weibull (weibull_min) - scale varies with time (loc=0 fixed to prevent threshold singularity)
    def nll_weibull(params):
        c, scale0, scale1_factor = params
        scale = scale0 * (1 + scale1_factor * t)
        ll = np.sum(stats.weibull_min.logpdf(data, c, loc=0.0, scale=scale))
        return -ll if np.isfinite(ll) else 1e10

    try:
        c_init, _, scale_init = stats.weibull_min.fit(data, floc=0)
        c_init = np.clip(c_init, 0.01, 50)
    except Exception:
        c_init, scale_init = 1.0, np.mean(data)

    bounds_weibull = [(0.01, 50), (0.001, None), (-0.99, 10.0)]
    res_weibull = optimize.minimize(nll_weibull, [c_init, max(scale_init, 0.001), 0.0], method='L-BFGS-B', bounds=bounds_weibull)
    if res_weibull.success:
        bic = 2 * res_weibull.fun + 3 * np.log(len(data))
        c, scale0, scale1_factor = res_weibull.x
        results['weibull_min_nn'] = {'bic': bic, 'params': {'c': c, 'loc': 0.0, 'scale': scale0 * (1 + scale1_factor * t)}, 'dist': 'weibull_min', 'type': 'nn'}

    # 2.6 Non-stationary Pearson Type III (pearson3) - scale varies with time
    def nll_pearson3(params):
        skew, loc, scale0, scale1_factor = params
        scale = scale0 * (1 + scale1_factor * t)
        ll = np.sum(stats.pearson3.logpdf(data, skew, loc=loc, scale=scale))
        return -ll if np.isfinite(ll) else 1e10

    try:
        skew_init, loc_init, scale_init = stats.pearson3.fit(data)
        skew_init = np.clip(skew_init, -3.0, 3.0)
    except Exception:
        skew_init, loc_init, scale_init = 0.0, np.mean(data), np.std(data)

    bounds_pearson3 = [(-5.0, 5.0), (None, None), (0.001, None), (-0.99, 10.0)]
    res_pearson3 = optimize.minimize(nll_pearson3, [skew_init, loc_init, max(scale_init, 0.001), 0.0], method='L-BFGS-B', bounds=bounds_pearson3)
    if res_pearson3.success:
        bic = 2 * res_pearson3.fun + 4 * np.log(len(data))
        skew, loc, scale0, scale1_factor = res_pearson3.x
        results['pearson3_nn'] = {'bic': bic, 'params': {'skew': skew, 'loc': loc, 'scale': scale0 * (1 + scale1_factor * t)}, 'dist': 'pearson3', 'type': 'nn'}

    # 2.7 Non-stationary Exponential (expon) - scale varies with time (loc=0 fixed to prevent threshold singularity)
    def nll_expon(params):
        scale0, scale1_factor = params
        scale = scale0 * (1 + scale1_factor * t)
        ll = np.sum(stats.expon.logpdf(data, loc=0.0, scale=scale))
        return -ll if np.isfinite(ll) else 1e10

    try:
        _, scale_init = stats.expon.fit(data, floc=0)
    except Exception:
        scale_init = np.mean(data)

    bounds_expon = [(0.001, None), (-0.99, 10.0)]
    res_expon = optimize.minimize(nll_expon, [max(scale_init, 0.001), 0.0], method='L-BFGS-B', bounds=bounds_expon)
    if res_expon.success:
        bic = 2 * res_expon.fun + 2 * np.log(len(data))
        scale0, scale1_factor = res_expon.x
        results['expon_nn'] = {'bic': bic, 'params': {'loc': 0.0, 'scale': scale0 * (1 + scale1_factor * t)}, 'dist': 'expon', 'type': 'nn'}

    # If all fail, fallback to a standard stationary gamma fit
    if not results:
        logger.warning(f"All fits failed for {column_name}. Falling back to stationary gamma.")
        try:
            a_init, loc_init, scale_init = stats.gamma.fit(data, floc=0)
        except Exception:
            a_init, loc_init, scale_init = 1.0, 0.0, np.mean(data)
        best_dist_name = 'gamma'
        best_params = {'a': a_init, 'loc': loc_init, 'scale': np.full_like(t, scale_init, dtype=float)}
        return best_dist_name, best_params, None, 0

    # --- 3. GoF Evaluation and Hybrid GoF-BIC Model Selection ---
    sorted_candidates = sorted(results.keys(), key=lambda k: results[k]['bic'])
    min_bic = results[sorted_candidates[0]]['bic']
    
    cand_eval = {}
    for key in sorted_candidates:
        cand = results[key]
        dist_obj = getattr(st, cand['dist'])
        params = cand['params']

        eval_data = np.copy(data)
        # Apply deterministic randomized quantile dithering for discrete duration ties
        if column_name == "duration":
            rng = np.random.default_rng(seed=42)
            dither = rng.uniform(-0.49, 0.49, size=len(eval_data))
            eval_data = eval_data + dither
            eval_data = np.maximum(eval_data, 0.1)  # Ensure positive domain constraint

        try:
            u_t = dist_obj.cdf(eval_data, **params)
            u_t = np.clip(u_t, 1e-6, 1 - 1e-6)
            res = st.goodness_of_fit(
                st.uniform, u_t, known_params={'loc': 0, 'scale': 1}, n_mc_samples=500, statistic='ad'
            )
            p_val = res.pvalue if np.isfinite(res.pvalue) else 0.0
            cand_eval[key] = {'p_val': p_val, 'statistic': res.statistic, 'bic': cand['bic']}
        except Exception as e:
            logger.debug(f"GoF evaluation failed for {key}: {e}")
            cand_eval[key] = {'p_val': 0.0, 'statistic': np.nan, 'bic': cand['bic']}

    # 3-Tier Hierarchical GoF-then-BIC Model Selection
    # Tier 1: Select model with lowest BIC among those achieving statistical significance (p > 0.05)
    significant_cands = [k for k in sorted_candidates if cand_eval[k]['p_val'] > 0.05]
    acceptable_cands = [k for k in sorted_candidates if 0.01 < cand_eval[k]['p_val'] <= 0.05]

    if significant_cands:
        selected_key = significant_cands[0]
        if selected_key != sorted_candidates[0]:
            logger.info(
                f"Hierarchical Selection (Tier 1): Promoted {selected_key} (p={cand_eval[selected_key]['p_val']:.4f}, BIC={cand_eval[selected_key]['bic']:.2f}) "
                f"over rejected lowest-BIC model {sorted_candidates[0]} (p={cand_eval[sorted_candidates[0]]['p_val']:.4f}, BIC={cand_eval[sorted_candidates[0]]['bic']:.2f})"
            )
    elif acceptable_cands:
        selected_key = acceptable_cands[0]
        if selected_key != sorted_candidates[0]:
            logger.info(
                f"Hierarchical Selection (Tier 2): Selected acceptable model {selected_key} (p={cand_eval[selected_key]['p_val']:.4f}) "
                f"over rejected model {sorted_candidates[0]} (p={cand_eval[sorted_candidates[0]]['p_val']:.4f})"
            )
    else:
        # Tier 3: All models failed statistical hypothesis testing (p <= 0.01). Do not force fit; select lowest BIC and log poor fit warning.
        selected_key = sorted_candidates[0]
        logger.debug(f"Hierarchical Selection (Tier 3): All models scored p <= 0.01 for {column_name}. Accepting poor fit score without p-hacking.")

    winner = results[selected_key]
    best_dist_name = winner['dist']
    best_params = winner['params']
    p_val = cand_eval[selected_key]['p_val']
    
    logger.info(
        f"Selected distribution for {column_name} ({winner['type']}): {selected_key} (BIC: {winner['bic']:.2f}, GoF p-val: {p_val:.4f})"
    )

    if p_val > 0.05:
        logger.info(f"Fit is statistically significant (p={p_val:.4f})")
        fit_score = 2  # good fit
    elif p_val > 0.01:
        logger.warning(f"Weak fit (p={p_val:.4f}), but acceptable for large GCM sets.")
        fit_score = 1  # acceptable fit
    else:
        logger.warning(f"Poor fit (p={p_val:.4f}). Return periods may be unreliable.")
        fit_score = 0  # poor fit

    return best_dist_name, best_params, None, fit_score


def run_drought_copula_analysis(events_df, cutoff_year, copula_family="gumbel"):
    events_df = events_df.copy()
    valid_mask = np.isfinite(events_df['duration']) & (events_df['duration'] > 0) & np.isfinite(events_df['intensity']) & np.isfinite(events_df['start_year'])
    events_df = events_df[valid_mask].copy()
    events_df["intensity_abs"] = events_df["intensity"].abs()

    ref = events_df[events_df["start_year"] < cutoff_year]
    proj = events_df[events_df["start_year"] >= cutoff_year]

    periods = {"Reference": ref, "Projection": proj}
    results = {}

    family_map = {
        "gumbel": pv.BicopFamily.gumbel,
        "clayton": pv.BicopFamily.clayton,
        "frank": pv.BicopFamily.frank,
        "student-t": pv.BicopFamily.student,
        "t": pv.BicopFamily.student,
        "normal": pv.BicopFamily.gaussian,
        "joe": pv.BicopFamily.joe
    }

    for name, data in periods.items():
        if len(data) < 10:
            logger.warning(f"Warning: Not enough data for {name} period.")
            continue

        u_list = []
        for col in ["duration", "intensity_abs"]:
            params = stats.gamma.fit(data[col], floc=0)
            u_vars = stats.gamma.cdf(data[col], *params)
            u_list.append(u_vars)

        U = np.column_stack(u_list).astype(np.float64)

        try:
            fam_enum = family_map.get(copula_family.lower())
            if not fam_enum:
                raise ValueError(f"Unknown copula family: {copula_family}")
            
            cop = pv.Bicop(family=fam_enum)
            cop.fit(data=U)
            
            lam_u = get_upper_tail_dependence(cop)

            results[name] = {
                "Copula Family": copula_family.capitalize(),
                "Parameter": float(cop.parameters[0][0]) if cop.parameters.size > 0 else np.nan,
                "Kendall_Tau": float(cop.tau),
                "Upper_Tail_Dep": float(lam_u),
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
    # 1. Prepare Data and eliminate computational non-finite artifacts (e.g. inf intensities)
    events_df = events_df.copy()
    valid_mask = np.isfinite(events_df['duration']) & (events_df['duration'] > 0) & np.isfinite(events_df['intensity']) & np.isfinite(events_df['start_year'])
    events_df = events_df[valid_mask].copy()
    # events_df["intensity_abs"] = events_df["intensity"].abs()

    ref_data = events_df[(events_df["start_year"] >= ref_start_year) & (events_df["start_year"] <= ref_end_year)]
    proj_data = events_df[(events_df["start_year"] >= proj_start_year) & (events_df["start_year"] <= proj_end_year)]

    if len(ref_data) < 5 or len(proj_data) < 5:
        logger.warning(f"Insufficient data points for copula analysis (Ref events: {len(ref_data)}, Proj events: {len(proj_data)}). Skipping.")
        return pd.DataFrame()

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
        is_ref = (name == "Ref")
        for col in variables:
            logger.info(f'Fitting marginal distribution for {name} period, {col}..')
            best_dist_name, best_params, _, fit_score = select_best_drought_distribution(
                data, col, is_reference_period=is_ref
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
        u = np.clip(u, 0.00001, 0.99999)

        cop = None
        if copula_family is None:
            logger.info('Searching best copula family...')
            cop, _cop_rank_df, copula_family_selected = select_best_copula(u)
        else:
            copula_family_selected = copula_family
            _cop_rank_df = None
            logger.info(f'Fitting copula using {copula_family_selected} family...')
            family_map = {
                "gumbel": pv.BicopFamily.gumbel,
                "clayton": pv.BicopFamily.clayton,
                "frank": pv.BicopFamily.frank,
                "student-t": pv.BicopFamily.student,
                "t": pv.BicopFamily.student,
                "normal": pv.BicopFamily.gaussian,
                "joe": pv.BicopFamily.joe
            }
            try:
                fam_enum = family_map.get(copula_family.lower())
                cop = pv.Bicop(family=fam_enum)
                cop.fit(data=u)
            except Exception as e:
                logger.error(f"Error fitting {name} period: {e}")

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

        c_ref_arr = cop_ref.cdf(np.array([[u_ref, v_ref]]))
        c_ref_val = float(c_ref_arr[0])

        p_joint_ref = 1 - u_ref - v_ref + c_ref_val
        p_joint_ref = max(p_joint_ref, 1e-8)  # prevent division by zero
        t_joint_ref = min(1 / (p_joint_ref * e_ref), 10000.0)  # cap at 10,000 years

        # Joint Probability in Projection for the SAME thresholds
        u_proj = float(apply_dist(dist_info["Proj"]["duration"], "cdf", d_thresh))
        v_proj = float(apply_dist(dist_info["Proj"]["intensity"], "cdf", i_thresh))

        c_proj_arr = cop_proj.cdf(np.array([[u_proj, v_proj]]))
        c_proj_val = float(c_proj_arr[0])

        p_joint_proj = 1 - u_proj - v_proj + c_proj_val
        p_joint_proj = max(p_joint_proj, 1e-8)  # prevent division by zero
        t_joint_proj = min(1 / (p_joint_proj * e_proj), 10000.0)  # cap at 10,000 years

        likelihood_change = round(t_joint_ref / t_joint_proj, 2) if t_joint_proj > 0 else float('nan')

        comparison_results.append(
            {
                "Return period": T,
                "Hist_Duration_Thresh": round(d_thresh, 1),
                "Hist_Intensity_Thresh": round(i_thresh, 1),
                "Proj_Duration_Thresh": round(d_thresh_proj, 1),
                "Proj_Intensity_Thresh": round(i_thresh_proj, 1),
                "Ref_Joint_T": round(t_joint_ref, 1),
                "Proj_Joint_T": round(t_joint_proj, 1),
                "Likelihood_Change": likelihood_change,
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
        ref_df, column_name, is_reference_period=True
    )

    logger.info(f'Fitting marginal distribution for projected {column_name}...')
    best_dist_name_proj, best_params_proj, _, proj_fit_score = select_best_drought_distribution(
        proj_df, column_name, is_reference_period=False
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
    logger.info('Starting non-stationary copula analysis...')

    ref_start_year = config["ref_start_year"]
    ref_end_year = config["ref_end_year"]
    proj_end_year = config["proj_end_year"]  # analysis will be limited to the data before this year
    proj_start_year = config['proj_start_year']

    spatial_unit_col = config["spatial_unit_col"]
    spei_scales = config.get('spei_indices', [3, 12])

    copula_family = config.get('copula_family', 'gumbel')
    analysis_regions = config.get('spatial_units', [])
    exclude_month_events = config.get('exclude_month_events', True)  # exclude drought events of 1 month length --> create noise and are not necessarily drought events

    # use the whole year if growth season not provided
    growth_season_start = int(config.get('growth_season_start', 1))
    growth_season_end = int(config.get('growth_season_end', 12))
    growth_season_months = list(range(growth_season_start, growth_season_end + 1))

    return_periods = config.get('return_periods', [5, 10])
    ref_scenario = config.get('reference_scenario', 'historical')
    analysis_scenarios = config.get('analysis_scenarios', None)

    events_csv = config["dry_events_csv"]
    copula_csv = config.get('copula_analysis_csv', None)
    gcm_eval_csv = config.get('gcm_eval_csv', None)

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

    if analysis_regions is None or len(analysis_regions) == 0:
        spatial_units = df_events[spatial_unit_col].unique().tolist()
    else:
        spatial_units = analysis_regions  # limits the analysis to a subset of regions

    if analysis_scenarios is None or len(analysis_scenarios) == 0:
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

                        if not df_cop_analysis.empty:
                            df_cop_analysis["scenario"] = scenario
                            df_cop_analysis["gcm"] = gcm
                            df_cop_analysis["spei"] = spei_scale
                            df_cop_analysis[spatial_unit_col] = su
                            df_copuls_list.append(df_cop_analysis)
                    else:
                        logger.info(f"GCM {gcm} not fit for {spei_col} in {su}. Skipping the GCM.")


    df_copulas = pd.concat(df_copuls_list, ignore_index=True)

    # calculate reliability as sum of fit scores
    df_copulas["reliability"] = (
        df_copulas["Ref_duration_fit_score"]
        + df_copulas["Ref_intensity_fit_score"]
        + df_copulas["Proj_duration_fit_score"]
        + df_copulas["Proj_intensity_fit_score"]
    )
    os.makedirs(os.path.dirname(copula_csv), exist_ok=True)
    save_dataframe(df=df_copulas, csv_file=copula_csv)


if __name__ == "__main__":
    config_file = "../data/config_nut2.json"
    # config_file = "../data/config_ecoregions.json"

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