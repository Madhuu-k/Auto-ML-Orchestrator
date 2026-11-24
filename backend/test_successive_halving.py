from sklearn.model_selection import train_test_split

from data.dataset_loader import load_dataset
from search.successive_halving import SuccessiveHalving


# Load dataset
X, y = load_dataset(
    "../datasets/breast_cancer.csv",
    "diagnosis"
)

# Split into training and temporary data
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

# Split temporary data into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# Initialize Successive Halving
scheduler = SuccessiveHalving()

# Execute search
results = scheduler.run(
    X_train,
    y_train,
    X_val,
    y_val
)

print("\n========== FINAL SEARCH HISTORY ==========")

for stage in results["history"]:

    print("\nStage:", stage["stage"])

    print("Training samples:", stage["training_samples"])

    print("Promoted:", stage["promoted_candidates"])