
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

from utils.utils import load_config


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
    if val <= 1.5: return 'blue'
    elif val <= 2: return 'yellow'
    elif val <= 4: return 'orange'
    elif val <= 7: return 'red'
    else: return 'darkred'

# Define the alpha mapping function based on the 'cv' column (used as ci)
def get_alpha(val):
    if val <= 0.2: return 1.0
    elif val <= 0.5: return 0.7
    elif val <= 0.7: return 0.5
    elif val <= 0.9: return 0.3
    else: return 0.2

def plot_scenario_ri(gdf,
                     df_likelihood,
                     spei_scale,
                     return_interval,
                     sup_title,
                     spatial_unit_col,
                     scenarios_order,
                     file_path=None,
                     spatial_units_group_shapefile=None,
                     spatial_unit_group_col=None):

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


  # Load and prepare group shapefile if present
  gdf_group = None
  if spatial_units_group_shapefile is not None:
        gdf_group = gpd.read_file(spatial_units_group_shapefile)

  if gdf_group is not None:
      if spatial_unit_group_col is not None and spatial_unit_group_col in gdf_group.columns:
          gdf_group = gdf_group.to_crs(gdf_plot.crs)
      else:
          gdf_group = None

  # Create the plot
  fig, axes = plt.subplots(2, 2, figsize=(29, 18))
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

      # Add spatial grouping (e.g. countries) if present and label with group column
      if gdf_group is not None and spatial_unit_group_col in gdf_group.columns:
          gdf_group.plot(
              ax=ax,
              facecolor='none',
              edgecolor='black',
              linewidth=0.8,
              zorder=10
          )

          for idx, group_row in gdf_group.iterrows():
              val = group_row[spatial_unit_group_col]
              if pd.notna(val):
                  point = group_row['geometry'].representative_point()
                  ax.text(
                      x=point.x,
                      y=point.y,
                      s=str(val),
                      fontsize=20,
                      weight='bold',
                      ha='center',
                      va='center',
                      color='black',
                      zorder=11
                  )



      # Grid and titles
      ax.set_title(f'Scenario: {scenario}', fontsize=22, fontweight='bold')
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

  # Create 2D Bivariate Color-Alpha Matrix Legend
  # Adjust layout first to reserve bottom space for the 2D legend table
  plt.subplots_adjust(hspace=-0.05, wspace=0.05, top=0.9, bottom=0.2, left=0.05, right=0.95)

  # Dedicated legend axes at bottom center [left, bottom, width, height]
  ax_leg = fig.add_axes([0.32, 0.05, 0.36, 0.08])

  colors_list = ['blue', 'yellow', 'orange', 'red', 'darkred']
  alphas_list = [1, 0.7, 0.5, 0.3, 0.2]

  for i, col_val in enumerate(colors_list):
      for j, alpha_val in enumerate(alphas_list):
          rect = mpatches.Rectangle((j, i), 1, 1, facecolor=col_val, alpha=alpha_val, edgecolor='black', linewidth=0.5)
          ax_leg.add_patch(rect)

  ax_leg.set_xlim(0, len(alphas_list))
  ax_leg.set_ylim(0, len(colors_list))

  # Configure tick marks in the center of each cell
  ax_leg.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
  ax_leg.set_xticklabels(['<= 0.2', '0.2 - 0.5', '0.5 - 0.7', '0.7 - 0.9', '> 0.9'], fontsize=16)
  ax_leg.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
  ax_leg.set_yticklabels(['<= 1.5', '1.5 - 2', '2 - 4', '4 - 7', '> 7'], fontsize=16)

  ax_leg.set_xlabel("Coefficient of Variation", fontsize=18, fontweight='bold', labelpad=1)
  ax_leg.set_ylabel("Likelihood\n Change", fontsize=18, fontweight='bold', labelpad=6)
  ax_leg.tick_params(length=0)
  ax_leg.grid(False)

  plt.suptitle(sup_title, fontsize=22, y=0.95, weight='bold')
  
  
  if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      logger.info(f'{os.path.basename(file_path)} saved.')
  else:
      plt.show()

def plot_spei_maps(config):

    spatial_unit_col = config['spatial_unit_col']
    copula_aggregation_csv = config['copula_analysis_aggregation_csv']
    spatial_units_shapefile = config['spatial_units_shapefile']
    scenarios_order = config.get("scenarios_order", [])
    plots_dir = config.get('plots_dir', 'plots')
    home_dir = config.get('home_dir', '../Outputs')
    spatial_units_group_shapefile = config.get('spatial_units_group_shapefile', None)
    spatial_unit_group_col = config.get('spatial_unit_group_col_plot', None)

    plots_dir = os.path.join(home_dir, plots_dir, 'maps')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    gdf = gpd.read_file(spatial_units_shapefile)  # ecoregions/basins/nuts2 or other spatial units shapefile
    df = pd.read_csv(copula_aggregation_csv)  # copula gcm-aggregated results

    ri_list = config.get('return_periods_plots', [5, 10, 50])
    spei_scales = config.get('spei_indices', [3])

    if scenarios_order is None or len(scenarios_order) == 0:
        scenarios_order = df['scenario'].unique().tolist()
        scenarios_order.sort()

    for ri in ri_list:
        for spei_scale in spei_scales:
            sup_title = f'{ri} Years SPEI {spei_scale} Drought Likelihood Change'
            plot_file = f'{ri}_years_rp_{spei_scale}spei_drought_likelihood_change.png'
            plot_file = os.path.join(plots_dir, plot_file)
            plot_scenario_ri(gdf=gdf,
                             df_likelihood=df,
                             spei_scale=spei_scale,
                             return_interval=ri,
                             sup_title=sup_title,
                             spatial_unit_col = spatial_unit_col,
                             scenarios_order = scenarios_order,
                             file_path=plot_file,
                             spatial_units_group_shapefile = spatial_units_group_shapefile,
                             spatial_unit_group_col = spatial_unit_group_col
                             )


def plot_spei_dotplots_by_spatial_group(config):
    """
    Faceted grid dotplot of droughts RI likelihood change by a spatial unit group column (e.g. country).
    """

    spatial_unit_col = config['spatial_unit_col']
    spatial_unit_group_col = config.get('spatial_unit_group_col', None)
    copula_analysis_csv = config['copula_analysis_csv']
    return_periods_plots = config['return_periods_plots']
    plot_spatial_units = config.get('plot_spatial_units', [])
    spei_scales = config.get('spei_indices', [3])
    plots_dir = config.get('plots_dir', 'plots')
    home_dir = config.get('home_dir', '../Outputs')

    plots_dir = os.path.join(home_dir, plots_dir, 'dotplots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)


    df = pd.read_csv(copula_analysis_csv)  # copula results

    if spatial_unit_group_col and spatial_unit_group_col in df.columns.tolist():

        if plot_spatial_units is None or len(plot_spatial_units) == 0:
            plot_spatial_units = df[spatial_unit_col].unique().tolist()

            # TODO: debug (for fast plotting)
            # plot_spatial_units = random.sample(plot_spatial_units, 3)

        df = df[df[spatial_unit_col].isin(plot_spatial_units)]

        # convert to %
        #df['Likelihood_Change'] = round((df['Likelihood_Change'] - 1) * 100, 0)

        df.sort_values(['scenario', spatial_unit_col], inplace = True)
        spatial_unit_groups = df[spatial_unit_group_col].unique().tolist()

        for spei_scale in spei_scales:
            for spatial_unit_group in spatial_unit_groups:
                # Filter the DataFrame for the desired SPEI index and country
                logger.info(f'Plotting {spatial_unit_group}-SPEI{spei_scale} likelihood change...')
                df_filtered = df[(df['spei'] == spei_scale) &
                                 (df['Return period'].isin(return_periods_plots)) &
                                 (df[spatial_unit_group_col] == spatial_unit_group)
                ]

                plot_file = f'{spei_scale}_{spatial_unit_group}_spei_drought_change_dot_plot.png'
                plot_file = os.path.join(plots_dir, plot_file)

                sup_title = f'{spatial_unit_group.title()} SPEI{spei_scale} Drought Likelihood Change'

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
                g.fig.suptitle(sup_title, fontsize=16, fontweight='bold')

                # 3. Adjust spacing to prevent layout overlap
                g.fig.subplots_adjust(top=0.95)

                # Adjust layout and display the plot
                plt.tight_layout()

                if plot_file is not None:
                    plt.savefig(plot_file, bbox_inches='tight')
                    logger.info(f'{os.path.basename(plot_file)} saved.')
                else:
                    plt.show()
    else:
        logger.warning("Spatial grouping column not present in data or not set in config, skipping plot.")


def plot_spei_dotplots(config):

    spatial_unit_col = config['spatial_unit_col']
    copula_analysis_csv = config['copula_analysis_csv']
    return_periods_plots = config['return_periods_plots']
    plot_spatial_units = config.get('plot_spatial_units', [])
    spei_scales = config.get('spei_indices', [3])
    plots_dir = config.get('plots_dir', 'plots')

    home_dir = config.get('home_dir', '../Outputs')

    plots_dir = os.path.join(home_dir, plots_dir, 'dotplots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)


    df = pd.read_csv(copula_analysis_csv)  # copula results


    if plot_spatial_units is None or len(plot_spatial_units) == 0:
        plot_spatial_units = df[spatial_unit_col].unique().tolist()

        # TODO: debug (for fast plotting)
        # plot_spatial_units = random.sample(plot_spatial_units, 3)

    df = df[df[spatial_unit_col].isin(plot_spatial_units)]

    # convert to %
    #df['Likelihood_Change'] = round((df['Likelihood_Change'] - 1) * 100, 0)

    df.sort_values(['scenario', spatial_unit_col], inplace = True)

    for spei_scale in spei_scales:
        # Filter the DataFrame for the desired SPEI index
        df_filtered = df[(df['spei'] == spei_scale) & (df['Return period'].isin(return_periods_plots))]

        plot_file = f'{spei_scale}_spei_drought_change_dot_plot.png'
        plot_file = os.path.join(plots_dir, plot_file)

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

        if plot_file is not None:
            plt.savefig(plot_file, bbox_inches='tight')
            logger.info(f'{os.path.basename(plot_file)} saved.')
        else:
            plt.show()
        plt.close()





if __name__ == "__main__":

    # config_file_name = "config_ecoregions.json"
    # config_file_name = 'config_nut2.json'
    config_file_name = 'config_nut2_1_min.json'

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

        if config_dict.get("execute_map_plots", False):
            plot_spei_maps(config_dict)
    if config_dict.get('execute_dot_plots', False):
        #plot_spei_dotplots(config_dict)
        plot_spei_dotplots_by_spatial_group(config_dict)
