import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Paths
# =========================
script_dir = Path(__file__).resolve().parents[1]
excel_file = script_dir / "midasmultimodalimagedatasetforaibasedskincancer/release_midas.xlsx"
output_plot = script_dir / "data_analysis/midas_overview_figure.png"
output_locations = script_dir / "data_analysis/location_df.csv"

# =========================
# Load data
# =========================
data = pd.read_excel(excel_file)

# =========================
# Split data as in the R script
# =========================
data_classified = data[~data["midas_path"].isna()].copy()
data_unclassified = data[data["midas_path"].isna()].copy()

unclassified_notassed = data_unclassified[data_unclassified["clinical_impression_1"].isna()].copy()

# benign = 0, malignant = 1
data_classified["binary_label"] = np.where(
    data_classified["midas_path"].str.contains("malignant", case=False, na=False),
    1,
    0
)

data_classified["label_text"] = data_classified["binary_label"].map({
    0: "benign",
    1: "malignant"
})

benign = data_classified[data_classified["binary_label"] == 0].copy()
malignant = data_classified[data_classified["binary_label"] == 1].copy()

# =========================
# Console output as in the R script
# =========================
print("number of benign patients:", len(benign))
print("number of malignant patients:", len(malignant))
print("number of unclassified patients:", len(data_unclassified))
print("number of patients (all):", len(data))

locations = data_classified["midas_location"].dropna().unique()
print("number of different locations", len(locations))

location_df = pd.DataFrame(
    {
        "count_classified": [0] * len(locations),
        "count_unclassified": [0] * len(locations),
    },
    index=locations
)

for i, loc in enumerate(locations):
    location_data_classified = data_classified[data_classified["midas_location"] == loc]
    location_data_unclassified = data_unclassified[data_unclassified["midas_location"] == loc]

    location_df.iloc[i, 0] = len(location_data_classified)
    location_df.iloc[i, 1] = len(location_data_unclassified)

print(location_df)
location_df.to_csv(output_locations)

print("Number of patients:", data["midas_record_id"].nunique())

# =========================
# Prepare variables for plotting
# =========================

sex_counts = data["midas_gender"].value_counts().reindex(["female", "male"], fill_value=0)

data["midas_age"] = pd.to_numeric(data["midas_age"], errors="coerce")
data_classified["midas_age"] = pd.to_numeric(data_classified["midas_age"], errors="coerce")

label_counts = data_classified["label_text"].value_counts().reindex(["benign", "malignant"], fill_value=0)

location_counts = (
    data_classified["midas_location"]
    .value_counts()
    .sort_values(ascending=False)
    .head(10)
)

classification_counts = pd.Series({
    "Classified": len(data_classified),
    "Unclassified": len(data_unclassified)
})

# =========================
# Helper function for pie chart labels
# =========================
def autopct_with_n(values):
    total = np.sum(values)

    def inner(pct):
        n = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n(n = {n})"
    return inner

# =========================
# Create figure
# =========================
fig, axes = plt.subplots(2, 3, figsize=(24, 14), constrained_layout=True)
fig.suptitle("MIDAS Dataset Overview", fontsize=24, fontweight="bold", y=1.03)

# =========================
# Plot 1: Sex distribution
# =========================
wedges1, _, _ = axes[0, 0].pie(
    sex_counts.values,
    autopct=autopct_with_n(sex_counts.values),
    startangle=90,
    textprops={"fontsize": 11}
)
axes[0, 0].set_title("Sex Distribution", fontsize=15, pad=12)
axes[0, 0].legend(
    wedges1,
    [label for label in sex_counts.index],
    title="Sex",
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    fontsize=11,
    title_fontsize=11,
    frameon=False
)

# =========================
# Plot 2: Benign vs malignant
# =========================
wedges2, _, _ = axes[0, 1].pie(
    label_counts.values,
    autopct=autopct_with_n(label_counts.values),
    startangle=90,
    textprops={"fontsize": 11}
)
axes[0, 1].set_title("Benign vs Malignant", fontsize=15, pad=12)
axes[0, 1].legend(
    wedges2,
    [label.capitalize() for label in label_counts.index],
    title="Diagnosis",
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    fontsize=11,
    title_fontsize=11,
    frameon=False
)

# =========================
# Plot 3: Skin lesion positions
# =========================
location_counts = location_counts[0:10]
bars_loc = axes[0, 2].bar(location_counts.index, location_counts.values)
axes[0, 2].set_title("Skin Lesion Positions (10 most frequent ones)", fontsize=15, pad=12)
axes[0, 2].set_xlabel("Location", fontsize=12)
axes[0, 2].set_ylabel("Count", fontsize=12)
axes[0, 2].tick_params(axis="x", rotation=45, labelsize=10)
for label in axes[0, 2].get_xticklabels():
    label.set_horizontalalignment("right")

for bar in bars_loc:
    height = bar.get_height()
    axes[0, 2].text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=10
    )

# =========================
# Plot 4: Age distribution
# =========================
axes[1, 0].hist(data["midas_age"].dropna(), bins=20, edgecolor="black", alpha=0.7)
axes[1, 0].set_title("Age Distribution", fontsize=15, pad=12)
axes[1, 0].set_xlabel("Age", fontsize=12)
axes[1, 0].set_ylabel("Frequency", fontsize=12)

# =========================
# Plot 5: Age by diagnosis
# =========================
age_benign = data_classified[data_classified["label_text"] == "benign"]["midas_age"].dropna()
age_malignant = data_classified[data_classified["label_text"] == "malignant"]["midas_age"].dropna()

axes[1, 1].boxplot(
    [age_benign, age_malignant],
    labels=[
        f"Benign\n(n = {len(age_benign)})",
        f"Malignant\n(n = {len(age_malignant)})"
    ]
)
axes[1, 1].set_title("Age by Diagnosis", fontsize=15, pad=12)
axes[1, 1].set_ylabel("Age", fontsize=12)

# =========================
# Plot 6: Correlation with malignancy
# =========================
# =========================
# Plot 6: Metadata correlation with malignancy
# =========================
corr_df = data_classified.copy()

# Numeric features
corr_df["age"] = pd.to_numeric(corr_df["midas_age"], errors="coerce")
corr_df["length_mm"] = pd.to_numeric(corr_df["length_(mm)"], errors="coerce")
corr_df["width_mm"] = pd.to_numeric(corr_df["width_(mm)"], errors="coerce")
corr_df["area_mm2"] = corr_df["length_mm"] * corr_df["width_mm"]

metadata_cols = [
    "age",
    "length_mm",
    "width_mm",
    "area_mm2",
    "midas_gender",
    "midas_ethnicity",
    "midas_race",
    "midas_fitzpatrick",
    "midas_location",
    "midas_melanoma",
    "midas_distance",
]

corr_input = corr_df[metadata_cols + ["binary_label"]].copy()

# One-hot encode categorical metadata
corr_input = pd.get_dummies(
    corr_input,
    columns=[
        "midas_gender",
        "midas_ethnicity",
        "midas_race",
        "midas_fitzpatrick",
        "midas_location",
        "midas_melanoma",
        "midas_distance",
    ],
    drop_first=False
)

correlations = {}

for col in corr_input.columns:
    if col == "binary_label":
        continue

    valid = corr_input[[col, "binary_label"]].dropna()

    if len(valid) > 1 and valid[col].nunique() > 1:
        correlations[col] = valid[col].corr(valid["binary_label"])

corr_series = (
    pd.Series(correlations)
    .dropna()
    .sort_values(key=lambda x: x.abs(), ascending=False)
    .head(10)
    .sort_values()
)

# Clean labels for readability
clean_labels = (
    corr_series.index
    .str.replace("midas_", "", regex=False)
    .str.replace("_", " ", regex=False)
    .str.replace("gender ", "sex: ", regex=False)
    .str.replace("ethnicity ", "ethnicity: ", regex=False)
    .str.replace("race ", "race: ", regex=False)
    .str.replace("fitzpatrick ", "fitzpatrick: ", regex=False)
    .str.replace("location ", "location: ", regex=False)
    .str.replace("melanoma ", "melanoma history: ", regex=False)
    .str.replace("distance ", "distance: ", regex=False)
)

bars_corr = axes[1, 2].barh(clean_labels, corr_series.values)

axes[1, 2].axvline(0, color="black", linewidth=1)
axes[1, 2].set_title("Top Metadata Correlations with Malignancy", fontsize=15, pad=12)
axes[1, 2].set_xlabel("Pearson Correlation", fontsize=12)
axes[1, 2].tick_params(axis="y", labelsize=9)
axes[1, 2].tick_params(axis="x", labelsize=10)

for bar in bars_corr:
    width = bar.get_width()
    axes[1, 2].text(
        width,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.2f}",
        va="center",
        ha="left" if width >= 0 else "right",
        fontsize=9
    )

# =========================
# Save figure
# =========================
plt.savefig(output_plot, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Plot saved to: {output_plot}")
print(f"Location table saved to: {output_locations}")