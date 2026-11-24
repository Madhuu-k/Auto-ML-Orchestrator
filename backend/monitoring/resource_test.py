import time

from resource_monitor import ResourceMonitor

monitor = ResourceMonitor(sampling_interval=0.1)

monitor.start()

# Simulate CPU-intensive work
total = 0

for i in range(10_000_000):
    total += i

monitor.stop()

print(monitor.get_metrics())