import os
import copy
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import Rolling_Horizon_Allocation as rha


# ============================================================
# 1. SETTINGS
# ============================================================

METHODS = ["greedy", "nearest", "round_robin", "edd"]

ALPHA = 1.0
LOOKAHEAD_DAYS = 7
PENALTY = 10

PATIENTS_FILE = "patients.csv"
PROVIDERS_FILE = "providers.csv"

OUTPUT_DIR = "Provider_Stress_Output"
OLD_OUTPUT_DIRS = ["Provider_Pressure_Output", "Provider_Stress_Output"]

UNEQUAL_LOAD_SCENARIOS = {
    "balanced_60": {"HomecareA": 0.60, "HomecareB": 0.60, "HomecareC": 0.60},
    "A_overloaded": {"HomecareA": 0.90, "HomecareB": 0.60, "HomecareC": 0.60},
    "B_overloaded": {"HomecareA": 0.60, "HomecareB": 0.90, "HomecareC": 0.60},
    "C_overloaded": {"HomecareA": 0.60, "HomecareB": 0.60, "HomecareC": 0.90},
    "A_B_overloaded": {"HomecareA": 0.90, "HomecareB": 0.90, "HomecareC": 0.60},
    "all_high": {"HomecareA": 0.85, "HomecareB": 0.85, "HomecareC": 0.85},
    "crisis": {"HomecareA": 0.95, "HomecareB": 0.95, "HomecareC": 0.95},
}


# ============================================================
# 2. CLEAN OLD OUTPUT
# ============================================================

for folder in OLD_OUTPUT_DIRS:
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder)
            print(f"Deleted folder: {folder}")
        except PermissionError:
            print(f"Could not delete folder, file is probably open: {folder}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_provider_id(provider):
    for attr in ["provider_id", "hco_id", "id", "name"]:
        if hasattr(provider, attr):
            return getattr(provider, attr)
    raise AttributeError("Could not find provider ID attribute.")


def run_assignment(patients, providers, method, scenario_name):
    output = rha.rolling_horizon_assignment(
        patients=patients,
        providers=providers,
        alpha=ALPHA,
        lookahead_days=LOOKAHEAD_DAYS,
        method=method,
    )

    kpi = output["kpis"]

    row = {
        "scenario": scenario_name,
        "method": method,
        "alpha": ALPHA,
        "lookahead_days": LOOKAHEAD_DAYS,
        "penalty": PENALTY,
        "total_assigned": kpi["total_assigned"],
        "avg_distance_km": kpi["avg_distance_km"],
        "util_std_dev": kpi["utilization_std_dev_%"],
        "overcap_weeks_total": sum(kpi["overcapacity_weeks"].values()),
    }

    for hco, util in kpi["avg_utilization_%"].items():
        row[f"{hco}_avg_utilization"] = util

    for hco, weeks in kpi["overcapacity_weeks"].items():
        row[f"{hco}_overcap_weeks"] = weeks

    return row


# ============================================================
# 4. LOAD DATA
# ============================================================

base_patients = rha.load_patients_from_csv(PATIENTS_FILE)
base_providers = rha.load_providers_from_csv(PROVIDERS_FILE)

rha.OVERCAPACITY_PENALTY_WEIGHT = PENALTY

base_capacity = {
    get_provider_id(provider): provider.capacity_hrs_per_week
    for provider in base_providers
}


# ============================================================
# 5. RUN UNEQUAL LOAD ANALYSIS
# ============================================================

results = []

for scenario_name, load_profile in UNEQUAL_LOAD_SCENARIOS.items():
    for method in METHODS:
        patients = copy.deepcopy(base_patients)
        providers = copy.deepcopy(base_providers)

        for provider in providers:
            provider_id = get_provider_id(provider)

            if provider_id not in load_profile:
                raise KeyError(
                    f"Provider {provider_id} is not included in UNEQUAL_LOAD_SCENARIOS."
                )

            capacity = base_capacity[provider_id]
            provider.capacity_hrs_per_week = capacity
            provider.initial_load_hrs_per_week = capacity * load_profile[provider_id]

        results.append(
            run_assignment(
                patients=patients,
                providers=providers,
                method=method,
                scenario_name=scenario_name,
            )
        )

df = pd.DataFrame(results)


# ============================================================
# 6. EXPORT TABLES
# ============================================================

df.to_excel(os.path.join(OUTPUT_DIR, "Provider_Stress_Results.xlsx"), index=False)
df.to_csv(os.path.join(OUTPUT_DIR, "Provider_Stress_Results.csv"), index=False)

summary_by_method = (
    df.groupby("method")
    .agg({
        "util_std_dev": ["mean", "min", "max"],
        "avg_distance_km": ["mean", "min", "max"],
        "overcap_weeks_total": ["mean", "min", "max"],
        "total_assigned": ["mean", "min", "max"],
    })
    .round(2)
)

summary_by_method.to_excel(
    os.path.join(OUTPUT_DIR, "Summary_By_Method.xlsx")
)

summary_by_scenario = (
    df.groupby(["scenario", "method"])
    .agg({
        "util_std_dev": "mean",
        "avg_distance_km": "mean",
        "overcap_weeks_total": "mean",
    })
    .reset_index()
    .round(2)
)

summary_by_scenario.to_excel(
    os.path.join(OUTPUT_DIR, "Summary_By_Scenario.xlsx"),
    index=False,
)


# ============================================================
# 7. PLOTS
# ============================================================

def plot_bar_by_scenario(data, y_col, ylabel, title, filename):
    pivot = data.pivot(index="scenario", columns="method", values=y_col)

    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


plot_bar_by_scenario(
    summary_by_scenario,
    y_col="util_std_dev",
    ylabel="Utilization standard deviation (%)",
    title="Effect of unequal initial load on workload balance",
    filename="Unequal_Load_Utilization_Std.png",
)

plot_bar_by_scenario(
    summary_by_scenario,
    y_col="avg_distance_km",
    ylabel="Average distance per assignment (km)",
    title="Effect of unequal initial load on travel distance",
    filename="Unequal_Load_Avg_Distance.png",
)

plot_bar_by_scenario(
    summary_by_scenario,
    y_col="overcap_weeks_total",
    ylabel="Total overcapacity weeks",
    title="Effect of unequal initial load on overcapacity",
    filename="Unequal_Load_Overcapacity.png",
)


# ============================================================
# 8. PRINT OUTPUT
# ============================================================

print("\n=== Provider stress analysis finished ===")
print("\n=== Summary by method ===")
print(summary_by_method)

print("\nFiles created in folder:")
print(f"- {OUTPUT_DIR}")
print("\nFiles:")
print("- Provider_Stress_Results.xlsx")
print("- Provider_Stress_Results.csv")
print("- Summary_By_Method.xlsx")
print("- Summary_By_Scenario.xlsx")
print("- Unequal_Load_Utilization_Std.png")
print("- Unequal_Load_Avg_Distance.png")
print("- Unequal_Load_Overcapacity.png")