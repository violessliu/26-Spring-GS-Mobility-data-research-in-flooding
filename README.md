# 26-Spring-GS-Mobility-data-research-in-flooding


This repository contains the data and Python scripts for the project:

**Spatiotemporal Patterns and Drivers of CBG-Level Mobility Changes during the 2024 Los Angeles Flood**

The project analyzes Census Block Group (CBG)-level mobility changes during the February 2024 Los Angeles flood. It compares predicted baseline POI visits with actual observed POI visits during the flood period, then uses OLS and MGWR models to examine how local socioeconomic, demographic, physical, and transportation characteristics are associated with mobility change.

This workflow supports the report analysis from mobility prediction, mobility change mapping, OLS regression, MGWR spatial heterogeneity analysis, and dominant driver mapping. :contentReference[oaicite:0]{index=0}

---

## 1. Repository Structure

```text
26-Spring-GS-Mobility-data-research-in-flooding/
│
├── README.md
│
├── data/
│   └── ca_trajectories_with_all_vars_and_centroids.csv
│
├── mgwr_cbg_dataset.csv
├── mgwr_ols.py
├── loss_map.py
├── spatial heterogeneity analysis.py
└── dominant significant predictor.py
