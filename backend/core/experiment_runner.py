from sklearn.model_selection import train_test_split

from data.dataset_loader import load_dataset
from candidates.candidate_generator import generate_candidates
from training.model_trainer import train_model
from evaluation.model_evaluator import evaluate_model
from monitoring.resource_monitor import ResourceMonitor

from budget.budget_manager import BudgetManager
from core.config import config

def run_experiment(file_path, target_column):
    # Load dataset
    X, y = load_dataset(file_path, target_column)
    
    # Split for training and testing 
    # 60% Training, 40% Temporary (to be split into validation and testing)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    
    # Split temporary data into validation and test sets 
    # Total left over = 40% - split into 20% validation and 20% testing
    
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Initialize Budget Manager 
    budget_manager = BudgetManager(config["budget"])
    budget_manager.start()
    
    # Generate Candidates
    candidates = generate_candidates()
    
    # Store Results
    results = {}
    
    # Train and Evaluate each candidate
    for candidate_id, model in candidates.items():
        
        if not budget_manager.can_start_candidate():
            print(f"Stopping experiment due to budget constraints: {budget_manager.stop_reason}")
            break
        
        monitor = ResourceMonitor()
        monitor.start()
        
        try:
            trained_model = train_model(model, X_train, y_train)
        
        finally:
            monitor.stop()
        
        # Collect resource measurements
        resource_metrics = monitor.get_metrics()
        
        # Update the budget manager with the resource metrics
        budget_manager.record_candidate(resource_metrics)
        
        # Evaluate the model on the validation set
        metrics = evaluate_model(trained_model, X_val, y_val)
        
        results[candidate_id] = {
            "model" : type(model).__name__,
            "metrics": metrics,
            "resource_metrics": resource_metrics
        }
        
        # Stop if the latest candidate exceeded memory allowance
        if resource_metrics["peak_memory_mb"] >= config["budget"]["max_memory_mb"]:
            break
        
    # Return experiment results
    
    return {
        "dataset": file_path,
        "target_column": target_column,
        "split": {
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "testing_samples": len(X_test)
        },
        "candidates": results
    }
    
    
if __name__ == "__main__":
    
    results = run_experiment(
        file_path="../datasets/breast_cancer.csv",
        target_column="diagnosis"
    )

    print(results)