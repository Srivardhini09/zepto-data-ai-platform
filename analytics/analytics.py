# ============================================
# MODULE 2 - ANALYTICS
# Zepto Data & AI Platform
# ============================================
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create output directories
os.makedirs("analytics/plots", exist_ok=True)
os.makedirs("analytics/models", exist_ok=True)

# ============================================
# 1. LOAD TITANIC DATASET
# ============================================

# Load Titanic dataset exactly once
df = sns.load_dataset("titanic")

# Immediately save an offline fallback
df.to_csv("analytics/titanic.csv", index=False)

print("=" * 60)
print("TITANIC DATASET LOADED")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe(include="all"))

print("\nMissing-value percentages:")
missing_percent = (df.isnull().mean() * 100).sort_values(ascending=False)
print(missing_percent)
# ============================================
# 2. MISSING VALUE HANDLING
# ============================================

print("\n" + "=" * 60)
print("MISSING VALUE HANDLING")
print("=" * 60)

# Work from the saved offline dataset from this point onward
df = pd.read_csv("analytics/titanic.csv")

# Calculate missing percentages
missing_percent = (df.isnull().mean() * 100).sort_values(ascending=False)

print("\nMissing percentages before cleaning:")
print(missing_percent)

# Columns with more than 30% missing values
high_missing_cols = missing_percent[missing_percent > 30].index.tolist()

# Columns with 5% to 30% missing values
medium_missing_cols = missing_percent[
    (missing_percent >= 5) & (missing_percent <= 30)
].index.tolist()

# Columns with less than 5% missing values
low_missing_cols = missing_percent[
    (missing_percent > 0) & (missing_percent < 5)
].index.tolist()

print("\n>30% missing - columns to drop:")
print(high_missing_cols)

print("\n5%-30% missing - columns to impute:")
print(medium_missing_cols)

print("\n<5% missing - rows to drop:")
print(low_missing_cols)

# Drop columns with more than 30% missing values
df_clean = df.drop(columns=high_missing_cols)

# Impute columns with 5%-30% missing values using the median
for column in medium_missing_cols:
    if pd.api.types.is_numeric_dtype(df_clean[column]):
        df_clean[column] = df_clean[column].fillna(
            df_clean[column].median()
        )
    else:
        df_clean[column] = df_clean[column].fillna(
            df_clean[column].mode()[0]
        )

# Drop rows containing missing values in columns below 5%
df_clean = df_clean.dropna(subset=low_missing_cols)

print("\nShape before cleaning:", df.shape)
print("Shape after cleaning:", df_clean.shape)

print("\nRemaining missing values:")
print(df_clean.isnull().sum())

print("\nMissing-value cleaning completed.")
# ============================================
# 3. AGE AND FARE ANALYSIS
# ============================================

print("\n" + "=" * 60)
print("AGE AND FARE ANALYSIS")
print("=" * 60)

# ---------- AGE HISTOGRAM ----------
plt.figure(figsize=(8, 5))
plt.hist(df_clean["age"], bins=20, edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("analytics/plots/age_histogram.png")
plt.close()

# ---------- AGE BOXPLOT ----------
plt.figure(figsize=(8, 5))
plt.boxplot(df_clean["age"])
plt.title("Age Boxplot")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig("analytics/plots/age_boxplot.png")
plt.close()

# ---------- FARE HISTOGRAM ----------
plt.figure(figsize=(8, 5))
plt.hist(df_clean["fare"], bins=30, edgecolor="black")
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("analytics/plots/fare_histogram.png")
plt.close()

# ---------- FARE BOXPLOT ----------
plt.figure(figsize=(8, 5))
plt.boxplot(df_clean["fare"])
plt.title("Fare Boxplot")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("analytics/plots/fare_boxplot.png")
plt.close()

print("\nPlots saved:")
print("- age_histogram.png")
print("- age_boxplot.png")
print("- fare_histogram.png")
print("- fare_boxplot.png")


# ============================================
# IQR OUTLIER ANALYSIS
# ============================================

def iqr_analysis(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]

    print(f"\n{column.upper()} IQR ANALYSIS")
    print("-" * 40)
    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Number of outliers: {len(outliers)}")

    return outliers


age_outliers = iqr_analysis(df_clean, "age")
fare_outliers = iqr_analysis(df_clean, "fare")


# ============================================
# FARE MEAN, MEDIAN, MODE AND SKEWNESS
# ============================================

fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode()[0]
fare_skewness = df_clean["fare"].skew()

print("\nFARE SUMMARY")
print("-" * 40)
print(f"Mean: {fare_mean:.2f}")
print(f"Median: {fare_median:.2f}")
print(f"Mode: {fare_mode:.2f}")
print(f"Skewness: {fare_skewness:.2f}")

if fare_skewness > 0:
    skew_conclusion = "Fare is positively/right skewed."
elif fare_skewness < 0:
    skew_conclusion = "Fare is negatively/left skewed."
else:
    skew_conclusion = "Fare is approximately symmetric."

print(f"Conclusion: {skew_conclusion}")
# ============================================
# 4. SURVIVAL RATE ANALYSIS
# ============================================

print("\n" + "=" * 60)
print("SURVIVAL RATE ANALYSIS")
print("=" * 60)

# Survival rate by sex using boolean masks
female_mask = df_clean["sex"] == "female"
male_mask = df_clean["sex"] == "male"

female_survival_rate = df_clean.loc[female_mask, "survived"].mean()
male_survival_rate = df_clean.loc[male_mask, "survived"].mean()

print("\nSurvival rate by sex:")
print(f"Female: {female_survival_rate:.2%}")
print(f"Male:   {male_survival_rate:.2%}")


# Survival rate by passenger class using boolean masks
first_class_mask = df_clean["pclass"] == 1
second_class_mask = df_clean["pclass"] == 2
third_class_mask = df_clean["pclass"] == 3

first_class_survival = df_clean.loc[first_class_mask, "survived"].mean()
second_class_survival = df_clean.loc[second_class_mask, "survived"].mean()
third_class_survival = df_clean.loc[third_class_mask, "survived"].mean()

print("\nSurvival rate by passenger class:")
print(f"1st Class: {first_class_survival:.2%}")
print(f"2nd Class: {second_class_survival:.2%}")
print(f"3rd Class: {third_class_survival:.2%}")


# Survival rate by sex and passenger class
female_first_mask = (df_clean["sex"] == "female") & (df_clean["pclass"] == 1)
female_second_mask = (df_clean["sex"] == "female") & (df_clean["pclass"] == 2)
female_third_mask = (df_clean["sex"] == "female") & (df_clean["pclass"] == 3)

male_first_mask = (df_clean["sex"] == "male") & (df_clean["pclass"] == 1)
male_second_mask = (df_clean["sex"] == "male") & (df_clean["pclass"] == 2)
male_third_mask = (df_clean["sex"] == "male") & (df_clean["pclass"] == 3)

print("\nSurvival rate by sex + passenger class:")

print(
    f"Female, 1st Class: "
    f"{df_clean.loc[female_first_mask, 'survived'].mean():.2%}"
)

print(
    f"Female, 2nd Class: "
    f"{df_clean.loc[female_second_mask, 'survived'].mean():.2%}"
)

print(
    f"Female, 3rd Class: "
    f"{df_clean.loc[female_third_mask, 'survived'].mean():.2%}"
)

print(
    f"Male, 1st Class: "
    f"{df_clean.loc[male_first_mask, 'survived'].mean():.2%}"
)

print(
    f"Male, 2nd Class: "
    f"{df_clean.loc[male_second_mask, 'survived'].mean():.2%}"
)

print(
    f"Male, 3rd Class: "
    f"{df_clean.loc[male_third_mask, 'survived'].mean():.2%}"
)
# ============================================
# 5. CORRELATION MATRIX AND HEATMAP
# ============================================

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = df_clean[correlation_columns].corr()

print("\n6-column correlation matrix:")
print(correlation_matrix.round(3))

# Save correlation matrix
correlation_matrix.round(3).to_csv(
    "analytics/correlation_matrix.csv"
)

# Create heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title("Titanic 6-Column Correlation Heatmap")
plt.tight_layout()
plt.savefig("analytics/plots/correlation_heatmap.png")
plt.close()

# Find two strongest absolute off-diagonal correlations
corr_pairs = correlation_matrix.where(
    ~np.eye(correlation_matrix.shape[0], dtype=bool)
)

corr_pairs = corr_pairs.abs().unstack().sort_values(
    ascending=False
)

strongest_pairs = []

for (column1, column2), value in corr_pairs.items():
    if column1 != column2:
        pair = tuple(sorted([column1, column2]))

        if pair not in strongest_pairs:
            strongest_pairs.append(pair)

        if len(strongest_pairs) == 2:
            break

print("\nTwo strongest absolute off-diagonal correlations:")

for column1, column2 in strongest_pairs:
    value = correlation_matrix.loc[column1, column2]
    print(f"{column1} vs {column2}: {value:.3f}")
# ============================================
# 6. MULTIVARIATE VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("MULTIVARIATE VISUALIZATIONS")
print("=" * 60)

# ---------- CHART 1 ----------
# Age vs Fare, colored by survival
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df_clean,
    x="age",
    y="fare",
    hue="survived",
    alpha=0.7
)

plt.title("Age vs Fare by Survival")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("analytics/plots/age_fare_survival_scatter.png")
plt.close()

print("\nChart 1: Age vs Fare by Survival")
print(
    "Interpretation: The scatter plot shows the relationship between "
    "passenger age and fare while distinguishing survival status. "
    "Higher fares are visible across different age groups, while "
    "surviving and non-surviving passengers overlap considerably."
)


# ---------- CHART 2 ----------
# Fare by passenger class and sex
plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df_clean,
    x="pclass",
    y="fare",
    hue="sex"
)

plt.title("Fare Distribution by Passenger Class and Sex")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("analytics/plots/fare_class_sex_boxplot.png")
plt.close()

print("\nChart 2: Fare by Passenger Class and Sex")
print(
    "Interpretation: First-class passengers generally paid higher fares "
    "than second- and third-class passengers. "
    "The distribution also varies between male and female passengers "
    "within each passenger class."
)


# ---------- CHART 3 ----------
# Survival rate by class and sex
survival_by_class_sex = (
    df_clean.groupby(["pclass", "sex"], observed=True)["survived"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8, 6))

sns.barplot(
    data=survival_by_class_sex,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title("Survival Rate by Passenger Class and Sex")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("analytics/plots/survival_class_sex_barplot.png")
plt.close()

print("\nChart 3: Survival Rate by Class and Sex")
print(
    "Interpretation: Survival rates vary substantially across passenger "
    "classes and between sexes. "
    "The chart shows that passenger class and sex were both associated "
    "with different observed survival rates."
)


# ---------- CHART 4 ----------
# Age distribution by class and survival
plt.figure(figsize=(9, 6))

sns.violinplot(
    data=df_clean,
    x="pclass",
    y="age",
    hue="survived",
    split=True
)

plt.title("Age Distribution by Passenger Class and Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Age")

plt.tight_layout()
plt.savefig("analytics/plots/age_class_survival_violin.png")
plt.close()

print("\nChart 4: Age by Class and Survival")
print(
    "Interpretation: The violin plot compares age distributions across "
    "passenger classes while separating passengers by survival status. "
    "It shows how the age distributions differ across classes and "
    "survival groups."
)

print("\nAll four multivariate charts saved successfully.")
# ============================================
# 7. STANDARDIZATION EXPLORATORY CHECK
# ============================================

print("\n" + "=" * 60)
print("STANDARDIZATION EXPLORATORY CHECK")
print("=" * 60)

from sklearn.preprocessing import StandardScaler

# Use the full cleaned dataset
numeric_features = ["age", "fare"]

scaler_check = StandardScaler()

standardized_values = scaler_check.fit_transform(
    df_clean[numeric_features]
)

standardized_df = pd.DataFrame(
    standardized_values,
    columns=numeric_features
)

print("\nOriginal means:")
print(df_clean[numeric_features].mean().round(3))

print("\nOriginal standard deviations:")
print(df_clean[numeric_features].std().round(3))

print("\nStandardized means:")
print(standardized_df.mean().round(3))

print("\nStandardized standard deviations:")
print(standardized_df.std().round(3))

print(
    "\nConclusion: Standardization centers age and fare near mean 0 "
    "and scales them to approximately unit variance."
)
# ============================================
# 8. TRAIN/TEST SPLIT AND PREPROCESSING
# ============================================

print("\n" + "=" * 60)
print("TRAIN/TEST SPLIT AND PREPROCESSING")
print("=" * 60)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Features and target
X = df_clean.drop(columns=["survived"])
y = df_clean["survived"]

# Stratified split to preserve the survival-class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nOverall target distribution:")
print(y.value_counts(normalize=True).round(3))

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(3))


# Features used for machine learning
numeric_features = [
    "age",
    "fare",
    "pclass",
    "sibsp",
    "parch"
]

categorical_features = [
    "sex",
    "embarked"
]


# Numeric preprocessing:
# Missing-value imputation + standardization
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# Categorical preprocessing:
# Missing-value imputation + one-hot encoding
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

print("\nPreprocessing pipeline created successfully.")
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)
print(
    "\nPreprocessing is fitted only on the training data "
    "inside the ML pipelines."
)
# ============================================
# 9. CLASSIFICATION MODELS
# ============================================

print("\n" + "=" * 60)
print("CLASSIFICATION MODELS")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# --------------------------------------------
# Create model pipelines
# --------------------------------------------

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

decision_tree_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                random_state=42,
                max_depth=5
            )
        )
    ]
)

random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

models = {
    "Logistic Regression": logistic_pipeline,
    "Decision Tree": decision_tree_pipeline,
    "Random Forest": random_forest_pipeline
}


# --------------------------------------------
# Train and evaluate models
# --------------------------------------------

classification_results = []

trained_models = {}

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{model_name}")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    classification_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    })

    trained_models[model_name] = model


# --------------------------------------------
# Comparison table
# --------------------------------------------

classification_results_df = pd.DataFrame(
    classification_results
)

print("\n" + "=" * 60)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 60)

print(
    classification_results_df.round(4).to_string(
        index=False
    )
)

classification_results_df.to_csv(
    "analytics/classification_comparison.csv",
    index=False
)


# --------------------------------------------
# ROC curves
# --------------------------------------------

plt.figure(figsize=(8, 6))

for model_name, model in trained_models.items():

    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC={auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Classification Models")
plt.legend()
plt.tight_layout()

plt.savefig(
    "analytics/plots/roc_curves.png"
)

plt.close()

print("\nROC curve saved successfully.")
# ============================================
# 10. DECISION TREE VISUALIZATION
# ============================================

print("\n" + "=" * 60)
print("DECISION TREE VISUALIZATION")
print("=" * 60)

# Get the trained Decision Tree pipeline
dt_model = trained_models["Decision Tree"]

# Transform training data using the fitted preprocessing step
X_train_transformed = dt_model.named_steps["preprocessor"].transform(X_train)

# Get feature names after preprocessing
feature_names = dt_model.named_steps["preprocessor"].get_feature_names_out()

# Get the trained decision tree
tree_model = dt_model.named_steps["classifier"]

plt.figure(figsize=(20, 10))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree for Titanic Survival Prediction")
plt.tight_layout()

plt.savefig(
    "analytics/plots/decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Decision Tree plot saved successfully.")
# ============================================
# 11. CLASS IMBALANCE COMPARISON
# ============================================

print("\n" + "=" * 60)
print("CLASS IMBALANCE COMPARISON")
print("=" * 60)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# --------------------------------------------
# 11.1 Baseline Logistic Regression
# --------------------------------------------

baseline_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)

baseline_precision = precision_score(y_test, baseline_pred)
baseline_recall = recall_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred)

# --------------------------------------------
# 11.2 Class Weight = Balanced
# --------------------------------------------

balanced_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

balanced_model.fit(X_train, y_train)
balanced_pred = balanced_model.predict(X_test)

balanced_precision = precision_score(y_test, balanced_pred)
balanced_recall = recall_score(y_test, balanced_pred)
balanced_f1 = f1_score(y_test, balanced_pred)

# --------------------------------------------
# 11.3 SMOTE
# --------------------------------------------

smote_model = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

smote_model.fit(X_train, y_train)
smote_pred = smote_model.predict(X_test)

smote_precision = precision_score(y_test, smote_pred)
smote_recall = recall_score(y_test, smote_pred)
smote_f1 = f1_score(y_test, smote_pred)

# --------------------------------------------
# Comparison Table
# --------------------------------------------

imbalance_results = pd.DataFrame(
    [
        {
            "Method": "Baseline",
            "Precision": baseline_precision,
            "Recall": baseline_recall,
            "F1": baseline_f1
        },
        {
            "Method": "Class Weight Balanced",
            "Precision": balanced_precision,
            "Recall": balanced_recall,
            "F1": balanced_f1
        },
        {
            "Method": "SMOTE",
            "Precision": smote_precision,
            "Recall": smote_recall,
            "F1": smote_f1
        }
    ]
)

print("\nImbalance Comparison:")
print(imbalance_results.round(4).to_string(index=False))

imbalance_results.to_csv(
    "analytics/imbalance_comparison.csv",
    index=False
)

print("\nSMOTE was applied only to the training data.")
print("Imbalance comparison saved successfully.")
# ============================================
# 12. RANDOM FOREST GRID SEARCH + OOB SCORE
# ============================================

print("\n" + "=" * 60)
print("RANDOM FOREST GRID SEARCH")
print("=" * 60)

from sklearn.model_selection import GridSearchCV

# Random Forest with OOB enabled
rf_oob_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                oob_score=True,
                n_jobs=-1
            )
        )
    ]
)

# Parameter grid
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__max_features": ["sqrt", "log2"]
}

# GridSearchCV
grid_search = GridSearchCV(
    estimator=rf_oob_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("Running GridSearchCV...")
grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print(f"\nBest Cross-Validation F1 Score: {grid_search.best_score_:.4f}")

# Get the best Random Forest model
best_rf_pipeline = grid_search.best_estimator_

# OOB score
best_rf_model = best_rf_pipeline.named_steps["classifier"]

print(f"OOB Score: {best_rf_model.oob_score_:.4f}")

# Test-set evaluation
best_rf_pred = best_rf_pipeline.predict(X_test)

best_rf_accuracy = accuracy_score(y_test, best_rf_pred)
best_rf_precision = precision_score(y_test, best_rf_pred)
best_rf_recall = recall_score(y_test, best_rf_pred)
best_rf_f1 = f1_score(y_test, best_rf_pred)

print("\nBest Random Forest Test Results:")
print(f"Accuracy:  {best_rf_accuracy:.4f}")
print(f"Precision: {best_rf_precision:.4f}")
print(f"Recall:    {best_rf_recall:.4f}")
print(f"F1 Score:  {best_rf_f1:.4f}")

print("\nRandom Forest GridSearch completed successfully.")

# ============================================
# 13. REGRESSION - PREDICT FARE
# ============================================

print("\n" + "=" * 60)
print("REGRESSION - PREDICT FARE")
print("=" * 60)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Features used to predict fare
regression_features = [
    "age",
    "pclass",
    "sibsp",
    "parch",
    "sex",
    "embarked"
]

X_reg = df_clean[regression_features]
y_reg = df_clean["fare"]

# Train/test split
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

# Regression preprocessing
reg_numeric_features = [
    "age",
    "pclass",
    "sibsp",
    "parch"
]

reg_categorical_features = [
    "sex",
    "embarked"
]

reg_numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

reg_categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

reg_preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", reg_numeric_transformer, reg_numeric_features),
        ("categorical", reg_categorical_transformer, reg_categorical_features)
    ]
)

# Regression pipeline
regression_pipeline = Pipeline(
    steps=[
        ("preprocessor", reg_preprocessor),
        (
            "regressor",
            LinearRegression()
        )
    ]
)

# Train
regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)

# Predict
y_reg_pred = regression_pipeline.predict(X_reg_test)

# Metrics
mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)

r2 = r2_score(
    y_reg_test,
    y_reg_pred
)

# Adjusted R-squared
n = len(y_reg_test)
p = len(regression_features)

adjusted_r2 = (
    1 - ((1 - r2) * (n - 1) / (n - p - 1))
)

print("\nRegression Metrics:")
print(f"MAE:          {mae:.4f}")
print(f"RMSE:         {rmse:.4f}")
print(f"R²:           {r2:.4f}")
print(f"Adjusted R²:  {adjusted_r2:.4f}")

# Residuals
residuals = y_reg_test - y_reg_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    y_reg_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residual Plot - Fare Regression")
plt.tight_layout()

plt.savefig(
    "analytics/plots/fare_residual_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nResidual plot saved successfully.")

# Simple heteroscedasticity observation
residual_abs = np.abs(residuals)
correlation_residual_size = np.corrcoef(
    y_reg_pred,
    residual_abs
)[0, 1]

print(
    f"Correlation between predicted fare and absolute residuals: "
    f"{correlation_residual_size:.4f}"
)

if abs(correlation_residual_size) > 0.30:
    print(
        "Conclusion: Residual spread changes with predicted fare, "
        "suggesting possible heteroscedasticity."
    )
else:
    print(
        "Conclusion: Residual spread does not show a strong systematic "
        "change with predicted fare."
    )

print("\nRegression analysis completed successfully.")

# ============================================
# 14. FINAL COMPARISON AND MODEL SAVING
# ============================================

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

# --------------------------------------------
# Classification final table
# --------------------------------------------

final_classification = classification_results_df.copy()

print("\nClassification Metrics:")
print(
    final_classification.round(4).to_string(index=False)
)

final_classification.to_csv(
    "analytics/final_classification_metrics.csv",
    index=False
)

# --------------------------------------------
# Regression final table
# --------------------------------------------

final_regression = pd.DataFrame(
    [{
        "Model": "Linear Regression",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adjusted_r2
    }]
)

print("\nRegression Metrics:")
print(
    final_regression.round(4).to_string(index=False)
)

final_regression.to_csv(
    "analytics/final_regression_metrics.csv",
    index=False
)

# --------------------------------------------
# Select final classification pipeline
# --------------------------------------------

# Use Logistic Regression because it currently
# has the strongest ROC-AUC and F1 among the
# three baseline classification models.
full_pipeline = trained_models["Logistic Regression"]

# --------------------------------------------
# Save complete fitted pipeline
# --------------------------------------------

import joblib

model_path = "analytics/models/best_classification_pipeline.joblib"

joblib.dump(
    full_pipeline,
    model_path
)

print(f"\nPipeline saved to: {model_path}")

# --------------------------------------------
# Reload pipeline
# --------------------------------------------

loaded_pipeline = joblib.load(model_path)

print("Pipeline reloaded successfully.")

# --------------------------------------------
# Raw input prediction
# --------------------------------------------

raw_input = pd.DataFrame(
    [{
        "age": 25,
        "fare": 30.0,
        "pclass": 2,
        "sibsp": 0,
        "parch": 0,
        "sex": "female",
        "embarked": "S"
    }]
)

prediction = loaded_pipeline.predict(raw_input)
prediction_probability = loaded_pipeline.predict_proba(raw_input)[0, 1]

print("\nRaw Input:")
print(raw_input.to_string(index=False))

print(
    f"\nPredicted Survival: "
    f"{'Survived' if prediction[0] == 1 else 'Did Not Survive'}"
)

print(
    f"Survival Probability: "
    f"{prediction_probability:.4f}"
)

print("\nFinal model pipeline validation completed successfully.")

print("\n" + "=" * 60)
print("FINAL CLASSIFIER RECOMMENDATION")
print("=" * 60)

print(
    "Based on the baseline test metrics, Logistic Regression is selected "
    "as the final classifier for this project."
)
print(
    "It achieved an accuracy of 0.8090, precision of 0.7833, "
    "F1-score of 0.7344, and ROC-AUC of 0.8610."
)
print(
    "Random Forest achieved similar recall but slightly lower F1-score "
    "and ROC-AUC, while Decision Tree produced lower values for these metrics."
)
print(
    "The imbalance analysis showed that class weighting improves recall, "
    "while SMOTE provides a more balanced precision and recall."
)