import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =========================
# =========================
shp_path = r"D:\mobility_projects\tl_2024_06_bg\tl_2024_06_bg.shp"
coef_path = r"D:\mobility_projects\figure3_4_map_ready_local_coefficients.csv"

out_dir = Path(r"D:\mobility_projects\mgwr_maps_no_islands")
out_dir.mkdir(exist_ok=True)

big_output_path = out_dir / "figure3_all_variables_mgwr_coefficients_no_islands.png"

# =========================

# =========================
gdf = gpd.read_file(shp_path)
coef_df = pd.read_csv(coef_path)

gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(12)
coef_df["census_block_group"] = coef_df["census_block_group"].astype(str).str.zfill(12)


gdf = gdf[gdf["GEOID"].str.startswith("06037")].copy()


gdf_wgs = gdf.to_crs(epsg=4326)
gdf_wgs["rep_lat"] = gdf_wgs.geometry.representative_point().y
keep_geoids = gdf_wgs.loc[gdf_wgs["rep_lat"] > 33.55, "GEOID"]

gdf = gdf[gdf["GEOID"].isin(keep_geoids)].copy()


gdf = gdf.merge(
    coef_df,
    left_on="GEOID",
    right_on="census_block_group",
    how="left"
)

# =========================

# =========================
vars_list = [
    "income",
    "elderly_percentage",
    "relief_m",
    "road_density_km_per_km2",
    "population_density",
    "zero_vehicle_rate"
]

titles = {
    "income": "Income",
    "elderly_percentage": "Elderly Percentage",
    "relief_m": "Terrain Relief",
    "road_density_km_per_km2": "Road Density",
    "population_density": "Population Density",
    "zero_vehicle_rate": "Zero-Vehicle Rate"
}

# =========================

# =========================
def plot_variable_map(gdf, var, ax=None, save_path=None, is_single=False):
    coef_col = f"{var}_coef"
    t_col = f"{var}_t"
    sig_col = f"{var}_sig_95"

    plot_gdf = gdf.copy()
    plot_gdf["plot_value"] = plot_gdf[coef_col]

    if sig_col in plot_gdf.columns:
        plot_gdf.loc[plot_gdf[sig_col] == False, "plot_value"] = np.nan
        valid = plot_gdf[plot_gdf[sig_col] == True].copy()
    else:
        valid = plot_gdf.dropna(subset=[coef_col]).copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    plot_gdf.plot(
        column="plot_value",
        cmap="RdBu_r",
        linewidth=0.05,
        edgecolor="white",
        legend=True,
        ax=ax,
        missing_kwds={
            "color": "lightgrey",
            "label": "Not significant"
        }
    )

    if len(valid) > 0:
        coef_min = valid[coef_col].min()
        coef_max = valid[coef_col].max()
        t_min = valid[t_col].min()
        t_max = valid[t_col].max()

        title_text = (
            f"{titles[var]}\n"
            f"Coeff: {coef_min:.3f} to {coef_max:.3f}; "
            f"t-value: {t_min:.2f} to {t_max:.2f}"
        )
    else:
        title_text = f"{titles[var]}\nNo significant CBGs"

    ax.set_title(title_text, fontsize=12 if not is_single else 14)
    ax.axis("off")

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

# =========================

# =========================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for ax, var in zip(axes, vars_list):
    plot_variable_map(gdf, var, ax=ax, is_single=False)

plt.suptitle(
    "Figure 3. Spatial Distribution of MGWR Local Coefficients",
    fontsize=18,
    y=0.98
)

plt.tight_layout()
plt.savefig(big_output_path, dpi=300, bbox_inches="tight")
plt.show()

print("Big figure saved to:", big_output_path)

# =========================
# 6. 输出六张单独小图
# =========================
for var in vars_list:
    single_output_path = out_dir / f"{var}_mgwr_coefficient_tvalue_no_islands.png"
    plot_variable_map(
        gdf,
        var,
        save_path=single_output_path,
        is_single=True
    )
    print("Single figure saved to:", single_output_path)
