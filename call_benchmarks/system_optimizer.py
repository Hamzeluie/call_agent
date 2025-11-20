import psutil
import asyncio
from typing import Dict, List

class SystemOptimizer:
    @staticmethod
    async def check_system_resources() -> Dict:
        """Check if system has enough resources for high concurrency"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_free_gb": disk.free / (1024**3)
        }
    
    @staticmethod
    def optimize_system_limits():
        """Increase system limits for higher concurrency"""
        import resource
        import socket
        
        try:
            # Increase file descriptor limit
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(8192, hard), hard))
            print(f"File descriptor limit increased to: {min(8192, hard)}")
        except (ValueError, resource.error) as e:
            print(f"Could not increase file descriptor limit: {e}")
        
        # Optimize socket settings
        socket.setdefaulttimeout(30)

# Usage in your main orchestrator (b.txt)
async def optimize_for_concurrency():
    optimizer = SystemOptimizer()
    
    # Check resources
    resources = await optimizer.check_system_resources()
    print(f"System resources: {resources}")
    
    # Optimize limits
    optimizer.optimize_system_limits()
    
    # Calculate safe concurrency limit based on resources
    safe_concurrency = max(2, min(50, int((100 - resources['cpu_percent']) / 2)))
    print(f"Recommended safe concurrency: {safe_concurrency}")
    
    return safe_concurrency