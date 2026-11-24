import psutil

# memory is returned in bytes, so we convert it to megabytes (MB)
# MB = Bytes / (1024 * 1024)

def get_system_info():
    memory = psutil.virtual_memory()
    
    total_memory = memory.total / (1024 * 1024)
    available_memory = memory.available / (1024 * 1024)
    used_memory = memory.used / (1024 * 1024)
    
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    
    cpu_utilization = psutil.cpu_percent(interval=1)
    
    system_info = {
        "total_memory_mb": total_memory,
        "used_memory_mb": used_memory,
        "available_memory_mb": available_memory,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "cpu_utilization_percent": cpu_utilization
    }
    
    return system_info


print(get_system_info())