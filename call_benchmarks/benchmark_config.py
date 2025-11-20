import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any
import json

@dataclass
class BenchmarkConfig:
    # Concurrency settings
    max_concurrent_users: int = 10
    ramp_up_users: int = 2
    ramp_up_delay: float = 1.0
    
    # Test duration
    test_duration_seconds: int = 300
    warm_up_seconds: int = 30
    
    # Audio settings
    audio_files_dir: str = "./test_audio"
    audio_duration_seconds: int = 10
    
    # LLM settings
    sample_texts: List[str] = None
    
    # Metrics to track
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.sample_texts is None:
            self.sample_texts = [
                "Hello, I'd like to schedule an appointment for next week.",
                "Can you help me with my account balance?",
                "I need technical support for my recent purchase.",
                "What are your business hours?",
                "Could you transfer me to the billing department?"
            ]
        
        if self.metrics is None:
            self.metrics = ["ITL", "TPOT", "goodput", "TTFT", "throughput"]