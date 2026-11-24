from sklearn.model_selection import train_test_split

from data.dataset_loader import load_dataset
from baselines.flaml_baseline import run_flaml_baseline


# Load dataset
X, y = load_dataset(
    "../datasets/breast_cancer.csv",
    "diagnosis"
)

# Training + temporary split
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

# Validation + test split
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# Run FLAML
results = run_flaml_baseline(
    X_train,
    y_train,
    X_val,
    y_val,
    time_budget=30
)

print("\n========== FLAML BASELINE ==========")

print("Selected model:", results["model"])

print("Best configuration:", results["best_config"])

print("Validation loss:", results["best_validation_loss"])

print("Metrics:", results["metrics"])

print("Resources:", results["resources"])