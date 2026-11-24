from sklearn.base import clone
from sklearn.model_selection import train_test_split

from candidates.candidate_generator import generate_candidates
from training.model_trainer import train_model
from evaluation.model_evaluator import evaluate_model
from monitoring.resource_monitor import ResourceMonitor


class SuccessiveHalving:

    def __init__(self):

        self.fidelity_levels = [
            {
                "name": "low",
                "fraction": 0.2,
                "promote": 3
            },
            {
                "name": "medium",
                "fraction": 0.5,
                "promote": 2
            },
            {
                "name": "high",
                "fraction": 1.0,
                "promote": 2
            }
        ]

        self.history = []

    def create_fidelity_datasets(self, X_train, y_train):

        # Create the medium-fidelity dataset (50%)
        X_medium, _, y_medium, _ = train_test_split(
            X_train,
            y_train,
            train_size=0.5,
            random_state=42,
            stratify=y_train
        )

        # Create the low-fidelity dataset (20%)
        X_low, _, y_low, _ = train_test_split(
            X_medium,
            y_medium,
            train_size=0.4,
            random_state=42,
            stratify=y_medium
        )

        return {
            "low": (X_low, y_low),
            "medium": (X_medium, y_medium),
            "high": (X_train, y_train)
        }

    def run(self, X_train, y_train, X_val, y_val):

        candidates = generate_candidates()

        datasets = self.create_fidelity_datasets(
            X_train,
            y_train
        )

        active_candidates = candidates

        for stage in self.fidelity_levels:

            stage_name = stage["name"]

            X_stage, y_stage = datasets[stage_name]

            print(f"\n========== {stage_name.upper()} FIDELITY ==========")

            stage_results = {}

            for candidate_id, original_model in active_candidates.items():

                # Create a fresh model for this fidelity level
                model = clone(original_model)

                # Start resource monitoring
                monitor = ResourceMonitor()
                monitor.start()

                try:
                    trained_model = train_model(
                        model,
                        X_stage,
                        y_stage
                    )

                finally:
                    monitor.stop()

                # Evaluate on the same validation dataset
                metrics = evaluate_model(
                    trained_model,
                    X_val,
                    y_val
                )

                resources = monitor.get_metrics()

                stage_results[candidate_id] = {
                    "model": type(original_model).__name__,
                    "metrics": metrics,
                    "resources": resources
                }

                print(
                    candidate_id,
                    metrics["f1_score"]
                )

            # Rank candidates by validation F1
            ranked_candidates = sorted(
                stage_results.items(),
                key=lambda item: item[1]["metrics"]["f1_score"],
                reverse=True
            )

            # Determine how many candidates to promote
            promotion_count = stage["promote"]

            promoted_ids = [
                candidate_id
                for candidate_id, _ in ranked_candidates[:promotion_count]
            ]

            self.history.append({
                "stage": stage_name,
                "training_samples": len(X_stage),
                "results": stage_results,
                "promoted_candidates": promoted_ids
            })

            print("Promoted candidates:", promoted_ids)

            # Keep only promoted candidates
            active_candidates = {
                candidate_id: candidates[candidate_id]
                for candidate_id in promoted_ids
            }

        return {
            "history": self.history,
            "final_candidates": active_candidates
        }