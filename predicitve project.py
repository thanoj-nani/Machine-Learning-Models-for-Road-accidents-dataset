# ============================================================
# ROAD ACCIDENTS INDIA - MACHINE LEARNING PROJECT

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Professional plotting style
sns.set_theme(
    style="whitegrid",
    context="notebook",
    font_scale=1.0
)

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.frameon"] = True
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    log_loss,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

# ------------------------------------------------------------
# PROFESSIONAL OUTPUT HELPERS
# ------------------------------------------------------------

def section(number, title):
    print("\n")
    print("=" * 90)
    print(f"  SECTION {number}: {title.upper()}")
    print("=" * 90)


def subsection(title):
    print("\n" + "-" * 90)
    print(f"  {title}")
    print("-" * 90)


def show_table(title, table):
    subsection(title)
    print(table.to_string(index=False))


def finish_plot(title, xlabel=None, ylabel=None):
    """Format the current graph and display it only after all labels are set."""
    plt.title(title, fontsize=16, fontweight="bold", pad=15)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()
    plt.close()


def split_label(train_size):
    """Return clean train/test labels such as 80/20 and 90/10."""
    train_percent = round(train_size * 100)
    test_percent = 100 - train_percent
    return f"{train_percent}/{test_percent}"


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(r"C:\Users\THANOJ\Downloads\road_accidents_india_12k.csv")

section(1, "Dataset Loading & Initial Inspection")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe())


# ------------------------------------------------------------
# 3. BASIC EDA
# ------------------------------------------------------------

section(2, "Exploratory Data Analysis")

# Average values by state
print("\nAverage values by State:")
print(
    df.groupby("State")[
        ["Total_Accidents", "Fatal_Accidents", "Deaths", "Injuries"]
    ].mean().sort_values("Deaths", ascending=False).head(10)
)

# Total accidents by year
year_data = df.groupby("Year")[
    ["Total_Accidents", "Fatal_Accidents", "Deaths", "Injuries"]
].sum()

print("\nAccident totals by year:")
print(year_data)


# ------------------------------------------------------------
# 4. EDA GRAPHS
# ------------------------------------------------------------

sns.set_style("whitegrid")

# Graph 1 - Accidents by year
plt.figure(figsize=(11, 6))
plt.plot(
    year_data.index,
    year_data["Total_Accidents"],
    marker="o",
    linewidth=2.5,
    markersize=6,
    color="#E67E22"
)
finish_plot(
    "Annual Trend of Total Road Accidents",
    "Year",
    "Total Accidents"
)


# Graph 2 - Deaths by year
plt.figure(figsize=(11, 6))
plt.plot(
    year_data.index,
    year_data["Deaths"],
    marker="o",
    linewidth=2.5,
    markersize=6,
    color="#8E44AD"
)
finish_plot(
    "Annual Trend of Road Accident Deaths",
    "Year",
    "Deaths"
)


# Graph 3 - Top 10 states by deaths
state_deaths = (
    df.groupby("State")["Deaths"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(11, 6))
sns.barplot(
    x=state_deaths.values,
    y=state_deaths.index,
    orient="h",
    color="#2A9D8F"
)
finish_plot(
    "Top 10 States by Average Road Accident Deaths",
    "Average Deaths",
    "State"
)


# Graph 4 - Total accidents vs deaths
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="Total_Accidents",
    y="Deaths",
    alpha=0.45,
    s=45,
    color="#C0392B"
)
sns.regplot(
    data=df,
    x="Total_Accidents",
    y="Deaths",
    scatter=False,
    line_kws={"linewidth": 2, "color": "#F39C12"}
)
finish_plot(
    "Relationship Between Total Accidents and Deaths",
    "Total Accidents",
    "Deaths"
)


# Graph 5 - Distribution of deaths
plt.figure(figsize=(9, 6))
sns.histplot(
    df["Deaths"],
    bins=30,
    kde=True,
    color="#16A085"
)
finish_plot(
    "Distribution of Road Accident Deaths",
    "Deaths",
    "Frequency"
)


# ------------------------------------------------------------
# 5. CORRELATION
# ------------------------------------------------------------

section(3, "Correlation Analysis")

numeric_data = df.select_dtypes(include=np.number)

correlation = numeric_data.corr()

print(correlation)

plt.figure(figsize=(11, 8))
sns.heatmap(
    correlation,
    annot=True,
    cmap="YlOrBr",
    center=0,
    fmt=".2f",
    linewidths=0.5,
    square=True
)
finish_plot("Pearson Correlation Heatmap")


# ------------------------------------------------------------
# 6. REGRESSION PROBLEM
# ------------------------------------------------------------
#
# We will predict:
#       Deaths
#
# Target = Deaths
#
# We do NOT use Deaths as an input.
#
# State is converted to numbers using one-hot encoding.
# For simplicity, we use pandas get_dummies().
# ------------------------------------------------------------

section(4, "Regression Analysis")

regression_data = df.copy()

# Convert State into dummy variables
regression_data = pd.get_dummies(
    regression_data,
    columns=["State"],
    drop_first=True
)

X_reg = regression_data.drop("Deaths", axis=1)
y_reg = regression_data["Deaths"]


# ------------------------------------------------------------
# 7. FUNCTION FOR REGRESSION
# ------------------------------------------------------------

def run_regression(train_size):

    print("\n----------------------------------------")
    print("Regression train-test split:", split_label(train_size))
    print("----------------------------------------")

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg,
        y_reg,
        train_size=train_size,
        random_state=42
    )

    # Standardization for KNN and SVR.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------
    # Linear Regression / OLS
    # --------------------------------------------------------

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    linear_prediction = linear_model.predict(X_test)

    # --------------------------------------------------------
    # Polynomial Regression
    # --------------------------------------------------------
    #
    # Degree 2 polynomial regression.
    # We use the first important numerical variables to keep
    # the example simple and computationally reasonable.
    # --------------------------------------------------------

    poly_columns = [
        "Year",
        "Total_Accidents",
        "Fatal_Accidents",
        "Non_Fatal_Accidents",
        "Injuries",
        "Urban_Accidents",
        "Rural_Accidents"
    ]

    poly = PolynomialFeatures(
        degree=2,
        include_bias=False
    )

    X_train_poly = poly.fit_transform(
        X_train[poly_columns]
    )

    X_test_poly = poly.transform(
        X_test[poly_columns]
    )

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    poly_prediction = poly_model.predict(X_test_poly)

    # --------------------------------------------------------
    # Decision Tree
    # --------------------------------------------------------

    tree_model = DecisionTreeRegressor(
        max_depth=8,
        random_state=42
    )

    tree_model.fit(X_train, y_train)
    tree_prediction = tree_model.predict(X_test)

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    forest_model = RandomForestRegressor(
        n_estimators=40,
        random_state=42
    )

    forest_model.fit(X_train, y_train)
    forest_prediction = forest_model.predict(X_test)

    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------

    knn_model = KNeighborsRegressor(n_neighbors=5)

    knn_model.fit(X_train_scaled, y_train)
    knn_prediction = knn_model.predict(X_test_scaled)

    # --------------------------------------------------------
    # SVR
    # --------------------------------------------------------

    svr_model = SVR(kernel="rbf")

    svr_model.fit(X_train_scaled, y_train)
    svr_prediction = svr_model.predict(X_test_scaled)

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    predictions = {
        "Linear Regression": linear_prediction,
        "Polynomial Regression": poly_prediction,
        "Decision Tree": tree_prediction,
        "Random Forest": forest_prediction,
        "KNN": knn_prediction,
        "SVR": svr_prediction
    }

    results = []

    for name, prediction in predictions.items():

        mse = mean_squared_error(
            y_test,
            prediction
        )

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(
                y_test,
                prediction
            ),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(
                y_test,
                prediction
            )
        })

    results = pd.DataFrame(results)

    print(results.to_string(index=False))

    return results


# ------------------------------------------------------------
# 8. REGRESSION FOR DIFFERENT TRAINING PERCENTAGES
# ------------------------------------------------------------

reg_75 = run_regression(0.75)
reg_80 = run_regression(0.80)
reg_90 = run_regression(0.90)



print("\n80/20 Regression Metrics Table:")
print(reg_80.round(4).to_string(index=False))

# ------------------------------------------------------------
# 9. REGRESSION COMPARISON
# ------------------------------------------------------------

subsection("Regression Model Comparison Across Train-Test Splits")

reg_75["Split"] = "75/25"
reg_80["Split"] = "80/20"
reg_90["Split"] = "90/10"

all_regression = pd.concat(
    [reg_75, reg_80, reg_90],
    ignore_index=True
)

print(all_regression.round(4).to_string(index=False))

print("\nNote: For regression, lower MAE/MSE/RMSE is better and higher R2 is better.")


# Plot R2 for different train-test splits
plt.figure(figsize=(12, 6))
sns.barplot(
    data=all_regression,
    x="Model",
    y="R2",
    hue="Split",
    errorbar=None,
    palette=["#E67E22", "#2A9D8F", "#8E44AD"]
)
plt.xticks(rotation=25)
finish_plot(
    "Regression Performance: R² Across Train-Test Splits",
    "Regression Model",
    "R² Score"
)


# Plot RMSE for different splits
plt.figure(figsize=(12, 6))
sns.barplot(
    data=all_regression,
    x="Model",
    y="RMSE",
    hue="Split",
    errorbar=None,
    palette=["#E67E22", "#2A9D8F", "#8E44AD"]
)
plt.xticks(rotation=25)
finish_plot(
    "Regression Performance: RMSE Across Train-Test Splits",
    "Regression Model",
    "RMSE"
)


# ------------------------------------------------------------
# 10. ACTUAL VS PREDICTED FOR 80/20
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

prediction = linear_model.predict(X_test)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    prediction,
    alpha=0.5
)

plt.xlabel("Actual Deaths")
plt.ylabel("Predicted Deaths")
plt.title("Actual vs Predicted Deaths - Linear Regression")

plt.show()
plt.close()


# Additional regression metrics
for metric, title in [
    ("MAE", "Regression MAE Across Train-Test Splits"),
    ("MSE", "Regression MSE Across Train-Test Splits")
]:
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=all_regression,
        x="Model",
        y=metric,
        hue="Split",
        errorbar=None
    )
    plt.xticks(rotation=25)
    finish_plot(title, "Regression Model", metric)


# ============================================================
# 11. CLASSIFICATION
# ============================================================
#
# The dataset does not contain a ready-made binary class.
#
# We create:
#
# Fatal_Rate = Fatal_Accidents / Total_Accidents
#
# High_Risk:
#       1 = fatal accident rate >= median
#       0 = fatal accident rate < median
#
# IMPORTANT:
# Fatal_Accidents is NOT used as an input because it directly
# creates the target. This prevents target leakage.
# ============================================================

section(5, "Classification Analysis")

classification_data = df.copy()

classification_data["Fatal_Rate"] = (
    classification_data["Fatal_Accidents"]
    / classification_data["Total_Accidents"]
)

median_rate = classification_data["Fatal_Rate"].median()

classification_data["High_Risk"] = (
    classification_data["Fatal_Rate"] >= median_rate
).astype(int)

# Convert State to dummy variables
classification_data = pd.get_dummies(
    classification_data,
    columns=["State"],
    drop_first=True
)

# Features
classification_features = [
    "Year",
    "Total_Accidents",
    "Non_Fatal_Accidents",
    "Injuries",
    "Urban_Accidents",
    "Rural_Accidents"
]

# Add state dummy variables
state_columns = [
    col for col in classification_data.columns
    if col.startswith("State_")
]

classification_features += state_columns

X_class = classification_data[
    classification_features
]

y_class = classification_data[
    "High_Risk"
]

print("\nClass distribution:")
print(y_class.value_counts())


# ------------------------------------------------------------
# 12. FUNCTION FOR CLASSIFICATION
# ------------------------------------------------------------

def run_classification(train_size):

    print("\n----------------------------------------")
    print("Classification train-test split:", split_label(train_size))
    print("----------------------------------------")

    X_train, X_test, y_train, y_test = train_test_split(
        X_class,
        y_class,
        train_size=train_size,
        random_state=42,
        stratify=y_class
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    logistic_model = LogisticRegression(
        max_iter=1000
    )

    logistic_model.fit(X_train_scaled, y_train)

    logistic_prediction = logistic_model.predict(
        X_test_scaled
    )

    logistic_probability = logistic_model.predict_proba(
        X_test_scaled
    )[:, 1]

    # --------------------------------------------------------
    # Decision Tree
    # --------------------------------------------------------

    tree_model = DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    )

    tree_model.fit(X_train, y_train)

    tree_prediction = tree_model.predict(X_test)

    tree_probability = tree_model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    forest_model = RandomForestClassifier(
        n_estimators=40,
        random_state=42
    )

    forest_model.fit(X_train, y_train)

    forest_prediction = forest_model.predict(X_test)

    forest_probability = forest_model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------

    knn_model = KNeighborsClassifier(
        n_neighbors=5
    )

    knn_model.fit(X_train_scaled, y_train)

    knn_prediction = knn_model.predict(
        X_test_scaled
    )

    knn_probability = knn_model.predict_proba(
        X_test_scaled
    )[:, 1]

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    # SVM is trained on standardized features. max_iter=-1 allows the solver to converge fully.
    svm_model = SVC(
        kernel="linear",
        probability=True,
        max_iter=-1,
        random_state=42
    )

    svm_model.fit(X_train_scaled, y_train)

    svm_prediction = svm_model.predict(
        X_test_scaled
    )

    svm_probability = svm_model.predict_proba(
        X_test_scaled
    )[:, 1]

    # --------------------------------------------------------
    # Predictions and probabilities
    # --------------------------------------------------------

    models = {
        "Logistic Regression": (
            logistic_prediction,
            logistic_probability
        ),
        "Decision Tree": (
            tree_prediction,
            tree_probability
        ),
        "Random Forest": (
            forest_prediction,
            forest_probability
        ),
        "KNN": (
            knn_prediction,
            knn_probability
        ),
        "SVM": (
            svm_prediction,
            svm_probability
        )
    }

    results = []

    for name, (prediction, probability) in models.items():

        cm = confusion_matrix(
            y_test,
            prediction
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(
                y_test,
                prediction
            ),
            "Log Loss": log_loss(
                y_test,
                probability
            ),
            "ROC-AUC": roc_auc_score(
                y_test,
                probability
            ),
            "Precision": precision_score(
                y_test,
                prediction,
                zero_division=0
            ),
            "Recall": recall_score(
                y_test,
                prediction,
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_test,
                prediction,
                zero_division=0
            )
        })

        # Confusion matrix
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Normal Risk", "High Risk"]
        ).plot(cmap="Oranges")

        plt.title(
            f"Confusion Matrix - {name} - "
            f"{split_label(train_size)}"
        )

        plt.show()
        plt.close()

    results = pd.DataFrame(results)

    print(results.to_string(index=False))

    return results, models, X_test, y_test


# ------------------------------------------------------------
# 13. CLASSIFICATION FOR 75/25, 80/20, 90/10
# ------------------------------------------------------------

class_75, models_75, X_test_75, y_test_75 = run_classification(0.75)

class_80, models_80, X_test_80, y_test_80 = run_classification(0.80)

class_90, models_90, X_test_90, y_test_90 = run_classification(0.90)



print("\n80/20 Classification Metrics Table:")
print(class_80.round(4).to_string(index=False))

# ------------------------------------------------------------
# 14. CLASSIFICATION COMPARISON
# ------------------------------------------------------------

section(6, "Model Comparison Across Train-Test Splits")

class_75["Split"] = "75/25"
class_80["Split"] = "80/20"
class_90["Split"] = "90/10"

all_classification = pd.concat(
    [class_75, class_80, class_90],
    ignore_index=True
)

print(
    all_classification.round(4).to_string(index=False)
)

print("\nNote: For classification, higher Accuracy/ROC-AUC/Precision/Recall/F1 is better, while lower Log Loss is better.")


# ------------------------------------------------------------
# 15. CLASSIFICATION METRIC GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

sns.barplot(
    data=all_classification,
    x="Model",
    y="F1 Score",
    hue="Split",
    palette=["#E67E22", "#2A9D8F", "#8E44AD"]
)

plt.xticks(rotation=30)
finish_plot("F1 Score for Different Train-Test Splits", "Model", "F1 Score")


# Accuracy comparison
plt.figure(figsize=(12, 6))

sns.barplot(
    data=all_classification,
    x="Model",
    y="Accuracy",
    hue="Split",
    palette=["#E67E22", "#2A9D8F", "#8E44AD"]
)

plt.xticks(rotation=30)
finish_plot("Accuracy for Different Train-Test Splits", "Model", "Accuracy")


# ROC-AUC comparison
plt.figure(figsize=(12, 6))

sns.barplot(
    data=all_classification,
    x="Model",
    y="ROC-AUC",
    hue="Split",
    palette=["#E67E22", "#2A9D8F", "#8E44AD"]
)

plt.xticks(rotation=30)
finish_plot("ROC-AUC for Different Train-Test Splits", "Model", "ROC-AUC")


# ------------------------------------------------------------
# 16. ROC CURVE - 80/20 SPLIT
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

roc_colors = ["#E67E22", "#2A9D8F", "#8E44AD", "#C0392B", "#D35400"]

for (name, (prediction, probability)), roc_color in zip(models_80.items(), roc_colors):

    fpr, tpr, thresholds = roc_curve(
        y_test_80,
        probability
    )

    auc = roc_auc_score(
        y_test_80,
        probability
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2.2,
        color=roc_color,
        label=f"{name} (AUC = {auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    color="#7F8C8D",
    label="Random Classifier"
)

plt.legend(loc="lower right")
finish_plot(
    "ROC Curves for Classification Models - 80/20 Split",
    "False Positive Rate",
    "True Positive Rate"
)


# ============================================================
# 17. FINAL CONCLUSION
# ============================================================

section(7, "Final Findings & Interpretation")

print("""
The project compares machine learning models using different
amounts of training data:

75% training / 25% testing
80% training / 20% testing
90% training / 10% testing

In general:

- More training data can help a model learn better patterns.
- However, the test set becomes smaller when using 90/10.
- Therefore, the metric can sometimes increase or decrease
  simply because the test sample changed.
- A model with the highest Accuracy is not always the best model.
- For classification, compare Precision, Recall, F1 and ROC-AUC
  together.
- For regression, lower MAE/MSE/RMSE and higher R2 are better.
- The 80/20 split is commonly used as a balanced choice.
- For a final academic project, it is useful to report results
  for all three splits rather than relying on only one split.
""")

print("\nPROJECT COMPLETED.")
