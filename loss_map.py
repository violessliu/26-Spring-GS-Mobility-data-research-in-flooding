import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =========================

# =========================
shp_path = r"D:\mobility_projects\tl_2024_06_bg\tl_2024_06_bg.shp"
csv_path = r"D:\mobility_projects\cbg_Yi_avg_loss_ratio_2024_0128_0131_vs_0204_0207.csv"
output_png = r"D:\mobility_projects\la_cbg_Yi_avg_loss_ratio_binned_mainland.png"

# =========================

# =========================
gdf = gpd.read_file(shp_path)
df = pd.read_csv(csv_path)

# =========================

# =========================
gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(12)
df["census_block_group"] = (
    df["census_block_group"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.zfill(12)
)


gdf_la = gdf[gdf["GEOID"].str.startswith("06037")].copy()

# =========================

# =========================
gdf_la = gdf_la.merge(
    df[["census_block_group", "Yi_avg_loss_ratio"]],
    left_on="GEOID",
    right_on="census_block_group",
    how="left"
)

# =========================

# =========================
gdf_la = gdf_la.to_crs(epsg=4326)

rep_points = gdf_la.representative_point()
gdf_la["rp_lon"] = rep_points.x
gdf_la["rp_lat"] = rep_points.y

mainland_mask = (
    (gdf_la["rp_lon"] > -119.05) &
    (gdf_la["rp_lon"] < -117.35) &
    (gdf_la["rp_lat"] > 33.62) &
    (gdf_la["rp_lat"] < 34.95)
)
gdf_la = gdf_la[mainland_mask].copy()

gdf_plot = gdf_la.explode(index_parts=False).copy()

rep_points2 = gdf_plot.representative_point()
gdf_plot["rp_lon"] = rep_points2.x
gdf_plot["rp_lat"] = rep_points2.y

mainland_mask2 = (
    (gdf_plot["rp_lon"] > -119.05) &
    (gdf_plot["rp_lon"] < -117.35) &
    (gdf_plot["rp_lat"] > 33.62) &
    (gdf_plot["rp_lat"] < 34.95)
)
gdf_plot = gdf_plot[mainland_mask2].copy()

# =========================

# =========================
bins = [-np.inf, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, np.inf]
labels = [
    "< -1.0",
    "-1.0 to -0.5",
    "-0.5 to -0.1",
    "-0.1 to 0",
    "0 to 0.1",
    "0.1 to 0.5",
    "0.5 to 1.0",
    "> 1.0"
]


color_map = {
    "< -1.0": "#d73027",
    "-1.0 to -0.5": "#f46d43",
    "-0.5 to -0.1": "#fddbc7",
    "-0.1 to 0": "#fee8e6",
    "0 to 0.1": "#e8f1fb",
    "0.1 to 0.5": "#d1e5f0",
    "0.5 to 1.0": "#92c5de",
    "> 1.0": "#4575b4"
}

gdf_la["Yi_bin"] = pd.cut(
    gdf_la["Yi_avg_loss_ratio"],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=True
)

gdf_plot["Yi_bin"] = pd.cut(
    gdf_plot["Yi_avg_loss_ratio"],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=True
)

# =========================

# =========================
bin_counts = (
    gdf_la.groupby("Yi_bin", observed=False)["GEOID"]
    .nunique()
    .reindex(labels)
    .fillna(0)
    .astype(int)
)


increase_count = gdf_la.loc[gdf_la["Yi_avg_loss_ratio"] < 0, "GEOID"].nunique()
decrease_count = gdf_la.loc[gdf_la["Yi_avg_loss_ratio"] > 0, "GEOID"].nunique()
zero_count = gdf_la.loc[gdf_la["Yi_avg_loss_ratio"] == 0, "GEOID"].nunique()

print("\n：")
print(bin_counts)

print("\n：")
print(f"（Yi < 0）: {increase_count}")
print(f"（Yi > 0）: {decrease_count}")
print(f"（Yi = 0）: {zero_count}")


count_df = pd.DataFrame({
    "Yi_bin": labels,
    "cbg_count": [bin_counts[label] for label in labels]
})
count_df.to_csv(
    r"D:\mobility_projects\la_cbg_Yi_bin_counts.csv",
    index=False,
    encoding="utf-8-sig"
)

summary_df = pd.DataFrame({
    "type": ["increase (Yi<0)", "decrease (Yi>0)", "no change (Yi=0)"],
    "cbg_count": [increase_count, decrease_count, zero_count]
})
summary_df.to_csv(
    r"D:\mobility_projects\la_cbg_Yi_summary_counts.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================

# =========================
fig, ax = plt.subplots(1, 1, figsize=(12, 12))


gdf_plot[gdf_plot["Yi_bin"].isna()].plot(
    ax=ax,
    color="lightgrey",
    edgecolor="white",
    linewidth=0.1
)


for label in labels:
    subset = gdf_plot[gdf_plot["Yi_bin"] == label]
    if len(subset) > 0:
        subset.plot(
            ax=ax,
            color=color_map[label],
            edgecolor="black",
            linewidth=0.08
        )


legend_handles = []
for label in labels:
    legend_handles.append(
        Patch(
            facecolor=color_map[label],
            edgecolor="black",
            label=f"{label} (n={bin_counts[label]})"
        )
    )

legend_handles.append(
    Patch(facecolor="lightgrey", edgecolor="white", label="No data")
)

ax.legend(
    handles=legend_handles,
    title=(
        "Yi_avg_loss_ratio\n"
        f"Increase (Yi<0): {increase_count}\n"
        f"Decrease (Yi>0): {decrease_count}"
    ),
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True
)

ax.set_title("LA County Mainland CBG Mobility Change (Binned Yi_avg_loss_ratio)", fontsize=15)
ax.axis("off")

plt.tight_layout()
plt.savefig(output_png, dpi=300, bbox_inches="tight")
plt.show()

print(f"\n: {output_png}")
print(": D:\\mobility_projects\\la_cbg_Yi_bin_counts.csv")
print(": D:\\mobility_projects\\la_cbg_Yi_summary_counts.csv")