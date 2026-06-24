import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import Rolling_Horizon_Allocation as rha


# ============================================================
# 1. SETTINGS
# ============================================================

ALPHAS = [i / 10 for i in range(0, 11)]
PENALTIES = [0, 1, 10, 20, 50]
METHODS = ["greedy", "nearest", "round_robin", "edd"]
ALPHA_SENSITIVE_METHODS = ["greedy", "edd"]

STANDARD_LOOKAHEAD_DAYS = 7

PATIENTS_FILE = "patients.csv"
PROVIDERS_FILE = "providers.csv"

OUTPUT_DIR = "Sensitivity_Output"


# ============================================================
# 2. CLEAN OLD OUTPUT
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in glob.glob(os.path.join(OUTPUT_DIR, "*")):
    try:
        os.remove(file)
    except PermissionError:
        print(f"Could not delete, file is probably open: {file}")


# ============================================================
# 3. LOAD DATA
# ============================================================

patients = rha.load_patients_from_csv(PATIENTS_FILE)
providers = rha.load_providers_from_csv(PROVIDERS_FILE)


# ============================================================
# 4. RUN SENSITIVITY ANALYSIS
# ============================================================

results = []

for method in METHODS:
    for alpha in ALPHAS:
        for penalty in PENALTIES:

            rha.OVERCAPACITY_PENALTY_WEIGHT = penalty

            output = rha.rolling_horizon_assignment(
                patients=patients,
                providers=providers,
                alpha=alpha,
                lookahead_days=STANDARD_LOOKAHEAD_DAYS,
                method=method
            )

            kpi = output["kpis"]

            row = {
                "method": method,
                "alpha": alpha,
                "lookahead_days": STANDARD_LOOKAHEAD_DAYS,
                "penalty": penalty,
                "total_assigned": kpi["total_assigned"],
                "avg_distance_km": kpi["avg_distance_km"],
                "util_std_dev": kpi["utilization_std_dev_%"],
                "overcap_weeks_total": sum(kpi["overcapacity_weeks"].values())
            }

            for hco, util in kpi["avg_utilization_%"].items():
                row[f"{hco}_avg_utilization"] = util

            for hco, weeks in kpi["overcapacity_weeks"].items():
                row[f"{hco}_overcap_weeks"] = weeks

            results.append(row)

df = pd.DataFrame(results)


# ============================================================
# 5. CREATE TABLES
# ============================================================

df.to_excel(os.path.join(OUTPUT_DIR, "Sensitivity_Analysis_Full.xlsx"), index=False)
df.to_csv(os.path.join(OUTPUT_DIR, "Sensitivity_Analysis_Full.csv"), index=False)

top10 = df.sort_values(
    by=["util_std_dev", "avg_distance_km", "total_assigned"],
    ascending=[True, True, False]
).head(10)

top10.to_excel(os.path.join(OUTPUT_DIR, "Top10_Configurations.xlsx"), index=False)

best_per_method = (
    df.sort_values(
        by=["method", "util_std_dev", "avg_distance_km", "total_assigned"],
        ascending=[True, True, True, False]
    )
    .groupby("method")
    .head(1)
    .reset_index(drop=True)
)

best_per_method.to_excel(os.path.join(OUTPUT_DIR, "Best_Per_Method.xlsx"), index=False)

summary_by_method = (
    df.groupby("method")
    .agg({
        "util_std_dev": ["mean", "min", "max"],
        "avg_distance_km": ["mean", "min", "max"],
        "overcap_weeks_total": ["mean", "min", "max"],
        "total_assigned": ["mean", "min", "max"]
    })
    .round(2)
)

summary_by_method.to_excel(os.path.join(OUTPUT_DIR, "Summary_By_Method.xlsx"))


# ============================================================
# 6. SENSITIVITY TABLES
# ============================================================

alpha_effect = (
    df[df["method"].isin(ALPHA_SENSITIVE_METHODS)]
    .groupby(["method", "alpha"])
    .agg({
        "util_std_dev": "mean",
        "avg_distance_km": "mean",
        "overcap_weeks_total": "mean"
    })
    .reset_index()
    .round(2)
)

penalty_effect = (
    df.groupby(["method", "penalty"])
    .agg({
        "util_std_dev": "mean",
        "avg_distance_km": "mean",
        "overcap_weeks_total": "mean"
    })
    .reset_index()
    .round(2)
)

alpha_effect.to_excel(os.path.join(OUTPUT_DIR, "Sensitivity_Alpha.xlsx"), index=False)
penalty_effect.to_excel(os.path.join(OUTPUT_DIR, "Sensitivity_Penalty.xlsx"), index=False)


# ============================================================
# 7. PLOTS
# ============================================================

plt.figure(figsize=(8, 5))
df.boxplot(column="util_std_dev", by="method")
plt.ylabel("Utilization standard deviation (%)")
plt.xlabel("Method")
plt.title("Distribution of load balancing performance per method")
plt.suptitle("")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Boxplot_Utilization_Std_Per_Method.png"), dpi=300)
plt.close()


plt.figure(figsize=(8, 5))

for method in ALPHA_SENSITIVE_METHODS:
    subset = alpha_effect[alpha_effect["method"] == method]
    plt.plot(
        subset["alpha"],
        subset["util_std_dev"],
        marker="o",
        label=method
    )

plt.xlabel("Alpha")
plt.ylabel("Average utilization standard deviation (%)")
plt.title("Sensitivity analysis: alpha")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Sensitivity_Alpha.png"), dpi=300)
plt.close()


plt.figure(figsize=(8, 5))

for method in METHODS:
    subset = penalty_effect[penalty_effect["method"] == method]
    plt.plot(
        subset["penalty"],
        subset["util_std_dev"],
        marker="o",
        label=method
    )

plt.xlabel("Overcapacity penalty")
plt.ylabel("Average utilization standard deviation (%)")
plt.title("Sensitivity analysis: overcapacity penalty")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Sensitivity_Penalty.png"), dpi=300)
plt.close()


# ============================================================
# 8. PRINT OUTPUT
# ============================================================

print("\n=== Best configuration per method ===")
print(best_per_method)

print("\n=== Top 10 configurations overall ===")
print(top10)

print("\n=== Summary by method ===")
print(summary_by_method)

print("\nFiles created in folder: Sensitivity_Output")
print("- Sensitivity_Analysis_Full.xlsx")
print("- Sensitivity_Analysis_Full.csv")
print("- Top10_Configurations.xlsx")
print("- Best_Per_Method.xlsx")
print("- Summary_By_Method.xlsx")
print("- Sensitivity_Alpha.xlsx")
print("- Sensitivity_Penalty.xlsx")
print("- Boxplot_Utilization_Std_Per_Method.png")
print("- Sensitivity_Alpha.png")
print("- Sensitivity_Penalty.png")