import warnings
warnings.filterwarnings("ignore")

import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

from mgwr.sel_bw import Sel_BW
from mgwr.gwr import MGWR


# =========================================================
# 0. 基本设置
# =========================================================
base_dir = Path(r"D:\mobility_projects")

# 使用你刚刚合并了 zero_vehicle_rate 的数据
input_file = base_dir / "mgwr_cbg_dataset_with_zero_vehicle_rate.csv"

# 输出文件
ols_coef_file = base_dir / "table3_ols_coefficients.csv"
vif_file = base_dir / "vif_table_with_zero_vehicle_rate.csv"

bw_file = base_dir / "mgwr_bandwidths.csv"
mgwr_local_file = base_dir / "figure3_4_mgwr_local_coefficients.csv"
mgwr_data_with_coef_file = base_dir / "mgwr_dataset_with_local_coefficients.csv"
model_summary_file = base_dir / "mgwr_model_summary.txt"

# Table 3 和 Table 4 专用输出
table3_file = base_dir / "table3_ols_results_for_paper.csv"
table4_file = base_dir / "table4_mgwr_results_for_paper.csv"

# Figure 3/4 专用输出
figure_map_file = base_dir / "figure3_4_map_ready_local_coefficients.csv"

# 因变量
y_col = "Yi_avg_loss_ratio"

# 自变量：保持原来的变量，只加 zero_vehicle_rate
x_cols = [
    "income",
    "elderly_percentage",
    "relief_m",
    "road_density_km_per_km2",
    "population_density",
    "zero_vehicle_rate",
]

# 坐标列
coord_cols = ["centroid_x_m", "centroid_y_m"]


# =========================================================
# 1. 读取数据
# =========================================================
df = pd.read_csv(input_file, low_memory=False)

df["census_block_group"] = (
    df["census_block_group"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.strip()
    .str.zfill(12)
)

df = df[df["census_block_group"].str.startswith("06037")].copy()

need_cols = ["census_block_group", y_col] + x_cols + coord_cols
missing_cols = [c for c in need_cols if c not in df.columns]
if missing_cols:
    raise KeyError(f"这些列不存在: {missing_cols}")

data = df[need_cols].copy()
data = data.dropna(subset=[y_col] + x_cols + coord_cols).copy()
data = data.reset_index(drop=True)

print("最终样本量:", len(data))
print("唯一 CBG 数:", data["census_block_group"].nunique())

print("\nY 描述统计:")
print(data[y_col].describe())


# =========================================================
# 2. OLS：Table 3
# =========================================================
X_ols_raw = data[x_cols].astype(float)
y_ols_raw = data[y_col].astype(float)

# 为了和论文 Table 3 更像，这里使用 standardized coefficient
X_ols_std = pd.DataFrame(
    StandardScaler().fit_transform(X_ols_raw),
    columns=x_cols
)

y_ols_std = pd.Series(
    StandardScaler().fit_transform(y_ols_raw.values.reshape(-1, 1)).flatten(),
    name=y_col
)

X_ols = sm.add_constant(X_ols_std)
ols_model = sm.OLS(y_ols_std, X_ols).fit(cov_type="HC3")

print("\n================ OLS SUMMARY ================\n")
print(ols_model.summary())

table3_df = pd.DataFrame({
    "variable": ols_model.params.index,
    "standardized_parameter": ols_model.params.values,
    "SE": ols_model.bse.values,
    "t_value": ols_model.tvalues.values,
    "p_value": ols_model.pvalues.values,
})

table3_df["significance"] = ""
table3_df.loc[table3_df["p_value"] < 0.10, "significance"] = "*"
table3_df.loc[table3_df["p_value"] < 0.05, "significance"] = "**"
table3_df.loc[table3_df["p_value"] < 0.01, "significance"] = "***"

table3_df.to_csv(table3_file, index=False, encoding="utf-8-sig")
table3_df.to_csv(ols_coef_file, index=False, encoding="utf-8-sig")

print("\nTable 3 OLS results:")
print(table3_df)

print("\nOLS R2:", ols_model.rsquared)
print("OLS AIC:", ols_model.aic)


# =========================================================
# 3. VIF
# =========================================================
vif_X = data[x_cols].astype(float).copy()
vif_X_const = sm.add_constant(vif_X)

vif_df = pd.DataFrame({
    "variable": vif_X_const.columns,
    "VIF": [
        variance_inflation_factor(vif_X_const.values, i)
        for i in range(vif_X_const.shape[1])
    ]
})

print("\n================ VIF ================\n")
print(vif_df)

vif_df.to_csv(vif_file, index=False, encoding="utf-8-sig")


# =========================================================
# 4. 准备 MGWR 数据
# =========================================================
coords = data[coord_cols].values.astype(float)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X = x_scaler.fit_transform(data[x_cols].values.astype(float))
y = y_scaler.fit_transform(data[[y_col]].values.astype(float))


# =========================================================
# 5. 选择 MGWR 带宽
# =========================================================
print("\n开始搜索 MGWR 带宽，这一步会比较慢...")

selector = Sel_BW(coords, y, X, multi=True, spherical=False)
mgwr_bw = selector.search()

print("\nMGWR 最优带宽:")
print(mgwr_bw)

param_names = ["Intercept"] + x_cols

bw_df = pd.DataFrame({
    "variable": param_names,
    "bandwidth": mgwr_bw
})

bw_df.to_csv(bw_file, index=False, encoding="utf-8-sig")


# =========================================================
# 6. 拟合 MGWR
# =========================================================
print("\n开始拟合 MGWR...")

mgwr_model = MGWR(coords, y, X, selector=selector, spherical=False)
mgwr_results = mgwr_model.fit()

print("\n================ MGWR SUMMARY ================\n")
mgwr_results.summary()


# =========================================================
# 7. 保存 summary
# =========================================================
ols_buf = io.StringIO()
with redirect_stdout(ols_buf):
    print(ols_model.summary())

mgwr_buf = io.StringIO()
with redirect_stdout(mgwr_buf):
    mgwr_results.summary()

with open(model_summary_file, "w", encoding="utf-8") as f:
    f.write("OLS SUMMARY\n")
    f.write(ols_buf.getvalue())
    f.write("\n\nVIF\n")
    f.write(vif_df.to_string(index=False))
    f.write("\n\nMGWR BANDWIDTHS\n")
    f.write(bw_df.to_string(index=False))
    f.write("\n\nMGWR SUMMARY\n")
    f.write(mgwr_buf.getvalue())


# =========================================================
# 8. 提取 MGWR local coefficients
# =========================================================
local_params = pd.DataFrame(
    mgwr_results.params,
    columns=param_names
)

local_tvals = pd.DataFrame(
    mgwr_results.filter_tvals(),
    columns=[f"{c}_t" for c in param_names]
)

mgwr_local_df = pd.concat(
    [
        data[["census_block_group"]].reset_index(drop=True),
        local_params.reset_index(drop=True),
        local_tvals.reset_index(drop=True),
    ],
    axis=1
)

mgwr_local_df.to_csv(mgwr_local_file, index=False, encoding="utf-8-sig")


# =========================================================
# 9. Table 4：MGWR mean / min / max / STD / bandwidth
# =========================================================
table4_df = pd.DataFrame({
    "variable": param_names,
    "mean": local_params[param_names].mean().values,
    "min": local_params[param_names].min().values,
    "max": local_params[param_names].max().values,
    "STD": local_params[param_names].std().values,
    "bandwidth": mgwr_bw,
})

# 计算每个变量显著 local coefficient 的比例
for var in param_names:
    t_col = f"{var}_t"
    if t_col in local_tvals.columns:
        table4_df.loc[table4_df["variable"] == var, "significant_count"] = (
            local_tvals[t_col].abs() > 1.96
        ).sum()
        table4_df.loc[table4_df["variable"] == var, "significant_ratio"] = (
            local_tvals[t_col].abs() > 1.96
        ).mean()

table4_df["MGWR_R2"] = mgwr_results.R2
table4_df["MGWR_AICc"] = mgwr_results.aicc

table4_df.to_csv(table4_file, index=False, encoding="utf-8-sig")

print("\n================ Table 4 MGWR results ================\n")
print(table4_df)


# =========================================================
# 10. Figure 3/4 地图所需文件
# 每个变量都有：
# coefficient
# filtered t-value
# significance flag
# =========================================================
figure_df = data.copy()

for var in param_names:
    figure_df[f"{var}_coef"] = local_params[var]
    figure_df[f"{var}_t"] = local_tvals[f"{var}_t"]
    figure_df[f"{var}_sig_95"] = figure_df[f"{var}_t"].abs() > 1.96

figure_df.to_csv(figure_map_file, index=False, encoding="utf-8-sig")


# =========================================================
# 11. 把局部系数并回总表
# =========================================================
output_df = pd.concat(
    [
        data.reset_index(drop=True),
        local_params.reset_index(drop=True),
        local_tvals.reset_index(drop=True),
    ],
    axis=1
)

output_df.to_csv(mgwr_data_with_coef_file, index=False, encoding="utf-8-sig")


# =========================================================
# 12. 输出结果
# =========================================================
print("\n================ 文件输出 ================\n")
print("Table 3 OLS 结果:", table3_file)
print("VIF 表:", vif_file)
print("MGWR 带宽表:", bw_file)
print("Table 4 MGWR 结果:", table4_file)
print("Figure 3/4 地图数据:", figure_map_file)
print("MGWR 局部系数表:", mgwr_local_file)
print("MGWR 带局部系数总表:", mgwr_data_with_coef_file)
print("模型摘要:", model_summary_file)

print("\n================ 当前模型设置 ================\n")
print("Y =", y_col)
print("X =", x_cols)
print("坐标 =", coord_cols)

print("\n完成。")