from flaml import AutoML

from monitoring.resource_monitor import ResourceMonitor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def run_flaml_baseline(
    X_train,
    y_train,
    X_val,
    y_val,
    time_budget=120
):

    # 1. Initialize FLAML
    automl = AutoML()

    # 2. Start resource monitoring
    monitor = ResourceMonitor()
    monitor.start()

    try:

        # 3. Run automated model search
        automl.fit(
            X_train=X_train,
            y_train=y_train,

            task="classification",
            metric="f1",

            time_budget=time_budget,

            estimator_list=[
                "lrl1",
                "lrl2",
                "rf",
                "extra_tree",
                "histgb"
            ],

            eval_method="holdout",

            X_val=X_val,
            y_val=y_val,

            seed=42,
            n_jobs=1,
            verbose=0
        )

    finally:
        monitor.stop()

    # 4. Get predictions through FLAML's API
    predictions = automl.predict(X_val)

    # 5. Calculate validation metrics
    metrics = {
        "accuracy": round(
            accuracy_score(y_val, predictions), 4
        ),

        "precision": round(
            precision_score(
                y_val,
                predictions,
                average="weighted",
                zero_division=0
            ), 4
        ),

        "recall": round(
            recall_score(
                y_val,
                predictions,
                average="weighted",
                zero_division=0
            ), 4
        ),

        "f1_score": round(
            f1_score(
                y_val,
                predictions,
                average="weighted",
                zero_division=0
            ), 4
        )
    }

    # 6. Collect resource measurements
    resource_metrics = monitor.get_metrics()

    # 7. Return results
    return {
        "model": automl.best_estimator,
        "best_config": automl.best_config,
        "best_validation_loss": automl.best_loss,
        "metrics": metrics,
        "resources": resource_metrics
    }