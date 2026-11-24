import psutil
import os
import time
import threading


class ResourceMonitor:

    def __init__(self, sampling_interval=0.05):

        self.process = psutil.Process(os.getpid())

        self.sampling_interval = sampling_interval

        self.start_time = None
        self.end_time = None

        self.start_cpu_times = None
        self.end_cpu_times = None

        self.monitoring = False
        self.monitor_thread = None

        self.cpu_samples = []
        self.memory_samples = []

        self.metrics = {}

    def start(self):

        self.cpu_samples = []
        self.memory_samples = []

        self.start_time = time.perf_counter()

        # Capture cumulative CPU time at the beginning
        self.start_cpu_times = self.process.cpu_times()

        self.monitoring = True

        # Start background monitoring
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources
        )

        self.monitor_thread.start()

    def _monitor_resources(self):

        while self.monitoring:

            # Periodic CPU measurement
            cpu_usage = self.process.cpu_percent(interval=None)

            # Process memory usage in MB
            memory_usage = (
                self.process.memory_info().rss / (1024 * 1024)
            )

            self.cpu_samples.append(cpu_usage)
            self.memory_samples.append(memory_usage)

            time.sleep(self.sampling_interval)

    def stop(self):

        self.monitoring = False

        if self.monitor_thread:
            self.monitor_thread.join()

        self.end_time = time.perf_counter()

        # Capture cumulative CPU time at the end
        self.end_cpu_times = self.process.cpu_times()

        execution_time = self.end_time - self.start_time

        # Calculate CPU time consumed
        start_cpu = (
            self.start_cpu_times.user +
            self.start_cpu_times.system
        )

        end_cpu = (
            self.end_cpu_times.user +
            self.end_cpu_times.system
        )

        cpu_time_used = end_cpu - start_cpu

        # Average CPU utilization across the entire execution
        average_cpu = (
            (cpu_time_used / execution_time) * 100
            if execution_time > 0 else 0
        )

        if self.memory_samples:

            average_memory = (
                sum(self.memory_samples) / len(self.memory_samples)
            )

            peak_memory = max(self.memory_samples)

        else:

            average_memory = 0
            peak_memory = 0

        # Peak CPU from periodic samples
        if len(self.cpu_samples) > 1:

            peak_cpu = max(self.cpu_samples)
            peak_cpu_estimated = False

        else:

            # Sampling was insufficient for a reliable peak
            peak_cpu = None
            peak_cpu_estimated = True

        self.metrics = {

            "execution_time_seconds": round(execution_time, 4),

            "average_cpu_percent": round(average_cpu, 2),

            "peak_cpu_percent": (
                round(peak_cpu, 2)
                if peak_cpu is not None else None
            ),

            "peak_cpu_estimated": peak_cpu_estimated,

            "average_memory_mb": round(average_memory, 2),

            "peak_memory_mb": round(peak_memory, 2),

            "cpu_samples": len(self.cpu_samples)

        }

    def get_metrics(self):

        return self.metrics