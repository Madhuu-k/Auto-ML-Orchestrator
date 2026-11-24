from pathlib import Path
from sklearn.model_selection import train_test_split

from data.dataset_loader import load_dataset
from search.successive_halving import SuccessiveHalving
from baselines.flaml_baseline import run_flaml_baseline
from monitoring.resource_monitor import ResourceMonitor


CANDIDATE_NAMES = {
    "C1": "Logistic Regression",
    "C2": "Random Forest",
    "C3": "Hist Gradient Boosting",
    "C4": "Decision Tree",
    "C5": "Extra Trees"
}


def print_table(headers, rows):

    widths = [
        max(len(str(row[i])) for row in [headers] + rows)
        for i in range(len(headers))
    ]

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    print(separator)

    print(
        "| " +
        " | ".join(
            str(headers[i]).ljust(widths[i])
            for i in range(len(headers))
        ) +
        " |"
    )

    print(separator)

    for row in rows:
        print(
            "| " +
            " | ".join(
                str(row[i]).ljust(widths[i])
                for i in range(len(headers))
            ) +
            " |"
        )

    print(separator)


def run_comparison(file_path, target_column):

    dataset_name = Path(file_path).stem

    print("\n" + "=" * 65)
    print("             BUDGET-AWARE AUTOML ORCHESTRATOR")
    print("=" * 65)

    print(f"\nDataset       : {dataset_name}")
    print(f"Target column : {target_column}")

    # STEP 1: Load dataset
    X, y = load_dataset(file_path, target_column)

    # STEP 2: Training and temporary split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    # STEP 3: Validation and test split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    print("\nDATASET PARTITION")

    print_table(
        ["Partition", "Samples"],
        [
            ["Training", len(X_train)],
            ["Validation", len(X_val)],
            ["Testing", len(X_test)]
        ]
    )

    # ==========================================
    # OUR MULTI-FIDELITY SYSTEM
    # ==========================================

    print("\n\n" + "=" * 65)
    print("              OUR MULTI-FIDELITY SYSTEM")
    print("=" * 65)

    scheduler = SuccessiveHalving()

    mfs_monitor = ResourceMonitor()
    mfs_monitor.start()

    try:
        mfs_results = scheduler.run(
            X_train,
            y_train,
            X_val,
            y_val
        )

    finally:
        mfs_monitor.stop()

    mfs_resources = mfs_monitor.get_metrics()

    # Display candidate performance at each fidelity
    for stage in mfs_results["history"]:

        print(f"\n{stage['stage'].upper()} FIDELITY")
        print(f"Training samples: {stage['training_samples']}")

        rows = []

        for candidate_id, result in stage["results"].items():

            rows.append([
                candidate_id,
                CANDIDATE_NAMES[candidate_id],
                result["metrics"]["accuracy"],
                result["metrics"]["f1_score"],
                result["resources"]["execution_time_seconds"]
            ])

        print_table(
            ["ID", "Candidate", "Accuracy", "F1-score", "Time (s)"],
            rows
        )

        print(
            "Promoted:",
            ", ".join(
                f"{cid} ({CANDIDATE_NAMES[cid]})"
                for cid in stage["promoted_candidates"]
            )
        )

    # Select best candidate from high fidelity
    final_stage = mfs_results["history"][-1]

    final_candidates = final_stage["results"]

    best_mfs_id, best_mfs_result = max(
        final_candidates.items(),
        key=lambda item: item[1]["metrics"]["f1_score"]
    )

    # ==========================================
    # FLAML BASELINE
    # ==========================================

    print("\n\n" + "=" * 65)
    print("                  FLAML BASELINE")
    print("=" * 65)

    flaml_results = run_flaml_baseline(
        X_train,
        y_train,
        X_val,
        y_val,
        time_budget=30
    )

    # ==========================================
    # FINAL COMPARISON
    # ==========================================

    print("\n\n" + "=" * 65)
    print("                  FINAL COMPARISON")
    print("=" * 65)

    print(f"\nDataset: {dataset_name}")

    our_metrics = best_mfs_result["metrics"]
    flaml_metrics = flaml_results["metrics"]

    our_resources = mfs_resources
    flaml_resources = flaml_results["resources"]

    comparison_rows = [

        [
            "Our MFS",
            CANDIDATE_NAMES[best_mfs_id],
            our_metrics["accuracy"],
            our_metrics["f1_score"],
            our_resources["execution_time_seconds"],
            our_resources["average_cpu_percent"],
            our_resources["peak_memory_mb"]
        ],

        [
            "FLAML",
            flaml_results["model"],
            flaml_metrics["accuracy"],
            flaml_metrics["f1_score"],
            flaml_resources["execution_time_seconds"],
            flaml_resources["average_cpu_percent"],
            flaml_resources["peak_memory_mb"]
        ]

    ]

    print_table(
        [
            "System",
            "Selected Model",
            "Accuracy",
            "F1-score",
            "Time (s)",
            "Avg CPU (%)",
            "Peak RAM (MB)"
        ],
        comparison_rows
    )

    print("\nEXPERIMENT SUMMARY")

    print(f"Dataset             : {dataset_name}")
    print(f"Total samples       : {len(X)}")
    print(f"Candidates initially: 5")
    print(f"Candidates at medium: 3")
    print(f"Candidates at high  : 2")
    print(f"MFS selected        : {CANDIDATE_NAMES[best_mfs_id]}")
    print(f"FLAML selected      : {flaml_results['model']}")

    return {
        "dataset": dataset_name,
        "our_system": {
            "selected_candidate": best_mfs_id,
            "selected_model": CANDIDATE_NAMES[best_mfs_id],
            "metrics": our_metrics,
            "resources": our_resources
        },
        "flaml": flaml_results
    }


if __name__ == "__main__":

    run_comparison(
        file_path="../datasets/breast_cancer.csv",
        target_column="diagnosis"
    )