import re

file_path = "/home/politti/git/speiApp/processing/copula_analysis_nn_station.py"
with open(file_path, "r") as f:
    content = f.read()

# 1. Imports
imports_old = r"""from copulae import \(
    GumbelCopula as OriginalGumbelCopula,
    FrankCopula,
    ClaytonCopula,
    StudentCopula,
    GumbelCopula,
    # JoeCopula,
    NormalCopula
\)
from fitter import Fitter
import scipy.stats as st
import scipy.optimize as optimize
from copulae import pseudo_obs.*?(?=\n\n# ---------------------------------------)"""

imports_new = """import pyvinecopulib as pv
from fitter import Fitter
import scipy.stats as st
import scipy.optimize as optimize

def get_upper_tail_dependence(cop, u=0.9999):
    c_val = cop.cdf(np.array([[u, u]]))
    return max(0.0, min((1 - 2*u + c_val[0]) / (1 - u), 1.0))
"""

content = re.sub(imports_old, imports_new, content, flags=re.DOTALL)

# 2. select_best_copula
select_old = r"""def select_best_copula\(df, columns=None\):.*?return best_fit\['obj'\], results_df, best_fit\['family'\]"""

select_new = """def select_best_copula(df, columns=None):
    \"\"\"
    Fits multiple copula families and selects the one with the lowest BIC using pyvinecopulib.
    \"\"\"
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
    return cop, None, cop.family.name"""

content = re.sub(select_old, select_new, content, flags=re.DOTALL)


# 3. run_drought_copula_analysis
run_old = r"""def run_drought_copula_analysis\(events_df, cutoff_year, copula_family="gumbel"\):.*?return pd.DataFrame\(results\)\.T"""

run_new = """def run_drought_copula_analysis(events_df, cutoff_year, copula_family="gumbel"):
    events_df = events_df.copy()
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

    return pd.DataFrame(results).T"""

content = re.sub(run_old, run_new, content, flags=re.DOTALL)


# 4. analyze_return_periods - replace get_copula
get_cop_old = r"""    def get_copula\(data, dist_info_subset\):.*?return cop, _cop_rank_df, copula_family_selected"""
get_cop_new = """    def get_copula(data, dist_info_subset):
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

        return cop, _cop_rank_df, copula_family_selected"""

content = re.sub(get_cop_old, get_cop_new, content, flags=re.DOTALL)


# 5. analyze_return_periods - fix CDF calls
cdf_ref_old = r"""        # Hack to bypass numpy 2\.0 TypeError in copulae's squeeze_output: pass array of length 2
        c_ref_arr = cop_ref\.cdf\(\[\[u_ref, v_ref\], \[u_ref, v_ref\]\]\)
        c_ref_val = float\(c_ref_arr\[0\]\)"""

cdf_ref_new = """        c_ref_arr = cop_ref.cdf(np.array([[u_ref, v_ref]]))
        c_ref_val = float(c_ref_arr[0])"""

content = re.sub(cdf_ref_old, cdf_ref_new, content, flags=re.DOTALL)

cdf_proj_old = r"""        c_proj_arr = cop_proj\.cdf\(\[\[u_proj, v_proj\], \[u_proj, v_proj\]\]\)
        c_proj_val = float\(c_proj_arr\[0\]\)"""

cdf_proj_new = """        c_proj_arr = cop_proj.cdf(np.array([[u_proj, v_proj]]))
        c_proj_val = float(c_proj_arr[0])"""

content = re.sub(cdf_proj_old, cdf_proj_new, content, flags=re.DOTALL)


with open(file_path, "w") as f:
    f.write(content)

print("Refactoring complete.")
