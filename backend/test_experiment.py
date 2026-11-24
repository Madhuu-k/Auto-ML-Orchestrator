from core.experiment_runner import run_experiment

results = run_experiment("data/sample.xlsx", "Purchased")

for candidate_id, result in results.items():
    print("\n", candidate_id)
    print("Model:", result["model"])
    print("Metrics:", result["metrics"])
    print("Resources:", result["resource_metrics"])