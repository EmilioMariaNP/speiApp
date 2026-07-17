
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import geopandas as gpd
from loguru import logger
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
import contextily as cx
import os
import numpy as np
import random

from utils.utils import load_config

# sspsrcp = ['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
# sspsrcp_labels =  ['Hist.', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
# sspsrcp_labels_dict = dict(zip(sspsrcp, sspsrcp_labels))
#
# exp_colours = ['blue','#3bb33b','#7303fc', '#ed7011','#971ad6']
# sspsrcp_colour_dict = dict(zip(sspsrcp_labels, exp_colours))

# uncertainty classes based on coefficient of variation
ci_uncert_classes = {0.3: 'low', 0.6: 'med', 0.8: 'high'}


def set_context(fig_size = (16,9), contx = 'talk', axes_style = 'darkgrid'):
    sns.set_style("darkgrid")
    sns.axes_style("darkgrid")
    sns.set_context(contx)
    sns.set_theme(rc={'figure.figsize':fig_size}) #width - height
    sns.axes_style(axes_style)
    return 0


set_context()



def plot_dists(
    df,
    fig_title,
    x_col="water_balance",
    hue_col="dataset",
    x_label="Water Balance (m)",
    out_file = None,
    show_fig = False
):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    sns.histplot(
        df,
        x=x_col,
        hue=hue_col,
        log_scale=True,
        element="step",
        fill=False,
        cumulative=True,
        stat="density",
        common_norm=False,
        ax=ax1,
    )
    sns.kdeplot(df, x=x_col, hue=hue_col, log_scale=True, ax=ax2, legend=False)
    fig.suptitle(fig_title)
    ax1.set_xlabel(x_label)
    ax2.set_xlabel(x_label)

    if show_fig:
        plt.show()

    if out_file is not None:
        plt.savefig(out_file)



def get_color(val):
    if val <= 1: return 'blue'
    elif val <= 2: return 'yellow'
    elif val <= 4: return 'orange'
    elif val <= 6: return 'red'
    else: return 'brown'

# Define the alpha mapping function based on the 'cv' column (used as ci)
def get_alpha(val):
    if val <= 0.2: return 1.0
    elif val <= 0.5: return 0.7
    elif val <= 0.7: return 0.5
    else: return 0.3

def plot_scenario_ri(gdf, df_likelihood, spei_scale, return_interval, sup_title, spatial_unit_col, scenarios_order):

    # Return period	scenario	spei	NUTS	Likelihood_Change	Likelihood_Change_std	Likelihood_Change_median	Likelihood_Change_cv

  # Filter the dataframe for the specific SPEI and RI requested
  df_geo_filtered = df_likelihood[(df_likelihood['spei'] == spei_scale) &
                                (df_likelihood['Return period'] == return_interval)
                                ].copy()

  # Merge with the geometry dataframe
  gdf_plot = gdf.merge(df_geo_filtered, on=spatial_unit_col, how='left')

  # Ensure the GDF is in Web Mercator for contextily
  gdf_plot = gdf_plot.to_crs(epsg=3857)

  # Apply mapping functions
  gdf_plot['color'] = gdf_plot['Likelihood_Change'].apply(get_color)
  gdf_plot['alpha'] = gdf_plot['Likelihood_Change_cv'].apply(get_alpha)

  # Create the plot
  fig, axes = plt.subplots(2, 2, figsize=(20, 18))
  axes_flat = axes.flatten()

  for i, scenario in enumerate(scenarios_order):
      ax = axes_flat[i]
      row, col = i // 2, i % 2
      scenario_gdf = gdf_plot[gdf_plot['scenario'] == scenario]

      # Plot each region separately to apply individual alpha values
      for _, row_data in scenario_gdf.iterrows():
          gpd.GeoSeries(row_data['geometry']).plot(
              ax=ax,
              color=row_data['color'],
              alpha=row_data['alpha'],
              edgecolor='black',
              linewidth=0.5
          )
      # Add basemap
      cx.add_basemap(ax, source=cx.providers.Esri.WorldPhysical)

      # Grid and titles
      ax.set_title(f'Scenario: {scenario}', fontsize=16, fontweight='bold')
      ax.grid(True, linestyle='--', alpha=0.6)

      # Disable exponential notation
      formatter = ScalarFormatter()
      formatter.set_scientific(False)
      formatter.set_useOffset(False)
      ax.xaxis.set_major_formatter(formatter)
      ax.yaxis.set_major_formatter(formatter)

      # Remove labels based on row/column position
      if row == 0:
          ax.set_xticklabels([])
      if col == 1:
          ax.set_yticklabels([])

  # Create Legend
  legend_elements = [
      mpatches.Patch(color='blue', label='mean <= 1'),
      mpatches.Patch(color='yellow', label='1 < mean <= 2'),
      mpatches.Patch(color='orange', label='2 < mean <= 4'),
      mpatches.Patch(color='red', label='4 < mean <= 6'),
      mpatches.Patch(color='brown', label='6 < mean')
  ]
  fig.legend(handles=legend_elements, loc='lower center', ncol=5, title="Likelihood Change Scale", fontsize=12)

  plt.suptitle(sup_title, fontsize=20, y=0.95)
  # Adjust layout and reduce vertical space between rows (hspace)
  plt.subplots_adjust(hspace=-0.005, wspace=0.1, top=0.9, bottom=0.1, left=0.05, right=0.95)
  plt.show()

def plot_spei_maps(config):

    spatial_unit_col = config['spatial_unit_col']
    copula_aggregation_csv = config['copula_analysis_aggregation_csv']
    spatial_units_shapefile = config['spatial_units_shapefile']
    scenarios_order = config_dict.get("scenarios_order", [])

    gdf = gpd.read_file(spatial_units_shapefile)  # ecoregions shapefile
    df = pd.read_csv(copula_aggregation_csv)  # copula gcm-aggregated results

    ri_list = [5, 10, 50]
    spei_scale = 3

    if scenarios_order is None or len(scenarios_order) == 0:
        scenarios_order = df['scenario'].unique().tolist()
        scenarios_order.sort()

    for ri in ri_list:
        sup_title = f'{ri} Years SPEI {spei_scale} Drought Likelihood Change'
        plot_scenario_ri(gdf=gdf,
                         df_likelihood=df,
                         spei_scale=spei_scale,
                         return_interval=ri,
                         sup_title=sup_title,
                         spatial_unit_col = spatial_unit_col,
                         scenarios_order = scenarios_order
                         )




def plot_sepi_dotplots(config):

    spatial_unit_col = config['spatial_unit_col']
    copula_analysis_csv = config['copula_analysis_csv']
    return_periods_plots = config['return_periods_plots']
    plot_spatial_units = config.get('plot_spatial_units', [])


    df = pd.read_csv(copula_analysis_csv)  # copula results

    if plot_spatial_units is None or len(plot_spatial_units) == 0:
        plot_spatial_units = df[spatial_unit_col].unique().tolist()

        # TODO: debug (for fast plotting)
        plot_spatial_units = random.sample(plot_spatial_units, 3)

    df = df[df[spatial_unit_col].isin(plot_spatial_units)]

    # convert to %
    #df['Likelihood_Change'] = round((df['Likelihood_Change'] - 1) * 100, 0)

    df.sort_values('scenario', inplace = True)


    # Filter the DataFrame for the desired SPEI index
    df_filtered = df[(df['spei'] == 3) & (df['Return period'].isin(return_periods_plots))]

    # Create the FacetGrid
    g = sns.FacetGrid(
        df_filtered,
        row=spatial_unit_col,
        col="scenario",
        height=4,
        aspect=1.5,
        margin_titles=True
    )

    # Map a line plot onto the grid, using median as the estimator
    g.map_dataframe(
        sns.pointplot,
        x="Return period",
        y="Likelihood_Change",
        estimator=np.median,
        errorbar='ci',
        linestyle='none',
        # join=False
    )

    # Set titles and labels for clarity
    g.set_axis_labels("Return Period", "Likelihood Change (Median)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.refline(y=1, color="red", linestyle="--")
    g.set(ylim=(-5, 50))

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":

    # config_file_name = "config_ecoregions.json"
    config_file_name = 'config_nut2.json'

    config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            config_file_name
        )
    config_dict = load_config(config_path)


    # update file paths
    home_dir = config_dict.get("home_dir", None)
    if home_dir:

        config_dict["dry_events_count_csv"] = os.path.join(
            home_dir, config_dict["dry_events_count_csv"]
        )
        config_dict["dry_events_csv"] = os.path.join(
            home_dir, config_dict["dry_events_csv"]
        )
        config_dict['copula_analysis_csv'] = os.path.join(home_dir, config_dict['copula_analysis_csv'])
        config_dict['spei_csv'] = os.path.join(home_dir, config_dict["spei_csv"])
        config_dict['copula_analysis_aggregation_csv'] = os.path.join(home_dir, config_dict['copula_analysis_aggregation_csv'])

    spatial_units_shapefile = config_dict.get('spatial_units_shapefile', None)

    if config_dict and spatial_units_shapefile:

        #plot_spei_maps(config_dict)
        plot_sepi_dotplots(config_dict)





















