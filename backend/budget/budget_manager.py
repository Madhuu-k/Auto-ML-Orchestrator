import time

class BudgetManager:
    def __init__(self, budget_config):

        self.max_runtime = budget_config["max_runtime_seconds"]
        self.max_memory = budget_config["max_memory_mb"]
        self.max_candidates = budget_config["max_candidates"]

        self.start_time = None

        self.candidates_evaluated = 0
        self.stop_reason = None

        self.peak_memory_mb = 0

    def start(self):

        self.start_time = time.perf_counter()

    def get_elapsed_time(self):

        if self.start_time is None:
            return 0

        return time.perf_counter() - self.start_time

    def can_start_candidate(self):

        if self.get_elapsed_time() >= self.max_runtime:
            self.stop_reason = "Runtime budget exceeded"
            return False

        if self.peak_memory_mb >= self.max_memory:
            self.stop_reason = "Memory budget exceeded"
            return False

        if self.candidates_evaluated >= self.max_candidates:
            self.stop_reason = "Candidate budget exhausted"
            return False

        return True

    def record_candidate(self, resource_metrics):

        self.candidates_evaluated += 1

        candidate_peak_memory = resource_metrics["peak_memory_mb"]

        self.peak_memory_mb = max(
            self.peak_memory_mb,
            candidate_peak_memory
        )

    def get_budget_status(self):

        return {
            "elapsed_time_seconds": round(self.get_elapsed_time(), 2),
            "candidates_evaluated": self.candidates_evaluated,
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "runtime_limit_seconds": self.max_runtime,
            "memory_limit_mb": self.max_memory,
            "candidate_limit": self.max_candidates,
            "stop_reason": self.stop_reason
        }