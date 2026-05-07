import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# =========================

# =========================
shp_path = r"D:\mobility_projects\tl_2024_06_bg\tl_2024_06_bg.shp"
coef_path = r"D:\mobility_projects\figure3_4_map_ready_local_coefficients.csv"

output_path = r"D:\mobility_projects\dominant_significant_mgwr_predictor_by_cbg_no_islands.png"

# =========================

# =========================
gdf = gpd.read_file(shp_path)
df = pd.read_csv(coef_path)

gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(12)
df["census_block_group"] = df["census_block_group"].astype(str).str.zfill(12)


gdf = gdf[gdf["GEOID"].str.startswith("06037")].copy()

# =========================

# =========================
gdf_wgs = gdf.to_crs(epsg=4326).copy()
gdf_wgs["rep_lat"] = gdf_wgs.geometry.representative_point().y


keep_geoids = gdf_wgs.loc[gdf_wgs["rep_lat"] > 33.55, "GEOID"]
gdf = gdf[gdf["GEOID"].isin(keep_geoids)].copy()

# =========================

# =========================
gdf = gdf.merge(
    df,
    left_on="GEOID",
    right_on="census_block_group",
    how="left"
)

# =========================

# =========================
variables = {
    "income": "Income",
    "elderly_percentage": "Elderly percentage",
    "relief_m": "Relief",
    "road_density_km_per_km2": "Road density",
    "population_density": "Population density",
    "zero_vehicle_rate": "Zero-vehicle rate",
}

# =========================

# =========================
dominant_input = pd.DataFrame(index=gdf.index)

for var in variables.keys():
    coef_col = f"{var}_coef"
    sig_col = f"{var}_sig_95"


    dominant_input[var] = np.where(
        gdf[sig_col] == True,
        gdf[coef_col].abs(),
        np.nan
    )


gdf["dominant_variable_key"] = dominant_input.idxmax(axis=1)

gdf.loc[dominant_input.isna().all(axis=1), "dominant_variable_key"] = None


gdf["dominant_variable"] = gdf["dominant_variable_key"].map(variables)

# =========================

# =========================
color_map = {
    "Income": "#ff7f0e",
    "Elderly percentage": "#9467bd",
    "Relief": "#d62728",
    "Road density": "#1f77b4",
    "Population density": "#2ca02c",
    "Zero-vehicle rate": "#17becf",
}

gdf["plot_color"] = gdf["dominant_variable"].map(color_map)

# =========================

# =========================
dominant_counts = gdf["dominant_variable"].value_counts(dropna=False)

print("\nDominant significant predictor counts:")
print(dominant_counts)

# =========================
#
# =========================
fig, ax = plt.subplots(figsize=(10, 14))


gdf.plot(
    ax=ax,
    color="#eeeeee",
    edgecolor="white",
    linewidth=0.05
)

# dominant significant predictor
for var, color in color_map.items():
    subset = gdf[gdf["dominant_variable"] == var]
    if len(subset) > 0:
        subset.plot(
            ax=ax,
            color=color,
            edgecolor="white",
            linewidth=0.05
        )

ax.set_title(
    "Dominant Significant MGWR Predictor by CBG\n"
    "Largest absolute standardized local coefficient among significant variables",
    fontsize=18
)

ax.axis("off")


handles = [
    mpatches.Patch(color=color, label=var)
    for var, color in color_map.items()
    if var in gdf["dominant_variable"].dropna().unique()
]


handles.append(
    mpatches.Patch(color="#eeeeee", label="No significant predictor")
)

ax.legend(
    handles=handles,
    title="Dominant significant variable",
    loc="lower left",
    fontsize=11,
    title_fontsize=12,
    frameon=True
)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved to:", output_path)
