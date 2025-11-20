import time
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional
import statistics
import json

@dataclass
class CallMetrics:
    call_id: str
    start_time: float
    end_time: Optional[float] = None
    first_audio_time: Optional[float] = None
    first_transcript_time: Optional[float] = None
    audio_chunks_sent: int = 0
    audio_chunks_received: int = 0
    total_audio_bytes_sent: int = 0
    total_audio_bytes_received: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class MetricsCollector:
    def __init__(self):
        self.active_calls: Dict[str, CallMetrics] = {}
        self.completed_calls: List[CallMetrics] = []
        self.start_time = time.time()
        
    def start_call(self, call_id: str) -> CallMetrics:
        metrics = CallMetrics(call_id=call_id, start_time=time.time())
        self.active_calls[call_id] = metrics
        return metrics
    
    def mark_first_audio(self, call_id: str):
        if call_id in self.active_calls:
            self.active_calls[call_id].first_audio_time = time.time()
    
    def mark_first_transcript(self, call_id: str):
        if call_id in self.active_calls:
            self.active_calls[call_id].first_transcript_time = time.time()
    
    def add_audio_sent(self, call_id: str, bytes_count: int):
        if call_id in self.active_calls:
            self.active_calls[call_id].audio_chunks_sent += 1
            self.active_calls[call_id].total_audio_bytes_sent += bytes_count
    
    def add_audio_received(self, call_id: str, bytes_count: int):
        if call_id in self.active_calls:
            self.active_calls[call_id].audio_chunks_received += 1
            self.active_calls[call_id].total_audio_bytes_received += bytes_count
    
    def end_call(self, call_id: str, error: str = None):
        if call_id in self.active_calls:
            metrics = self.active_calls[call_id]
            metrics.end_time = time.time()
            if error:
                metrics.errors.append(error)
            self.completed_calls.append(metrics)
            del self.active_calls[call_id]
    
    def calculate_metrics(self) -> Dict[str, Any]:
        if not self.completed_calls:
            return {}
        
        completed = [c for c in self.completed_calls if c.end_time]
        
        # ITL (Input Latency) - Time from user speech to agent response
        itl_times = []
        for call in completed:
            if call.first_audio_time and call.first_transcript_time:
                itl_times.append(call.first_transcript_time - call.first_audio_time)
        
        # TPOT (Time Per Output Token) - Not directly applicable to audio, using audio chunk latency
        tpot_times = []
        # This would require more granular audio processing timing
        
        # Goodput - Useful data transfer rate
        total_duration = sum(c.end_time - c.start_time for c in completed)
        total_useful_bytes = sum(c.total_audio_bytes_received for c in completed)
        goodput = total_useful_bytes / total_duration if total_duration > 0 else 0
        
        # TTFT (Time To First Token) - Time to first audio response
        ttft_times = []
        for call in completed:
            if call.first_audio_time:
                ttft_times.append(call.first_audio_time - call.start_time)
        
        # Throughput - Calls per second
        total_calls = len(completed)
        total_time = time.time() - self.start_time
        throughput = total_calls / total_time if total_time > 0 else 0
        
        return {
            "ITL": {
                "mean": statistics.mean(itl_times) if itl_times else 0,
                "median": statistics.median(itl_times) if itl_times else 0,
                "min": min(itl_times) if itl_times else 0,
                "max": max(itl_times) if itl_times else 0,
                "samples": len(itl_times)
            },
            "TTFT": {
                "mean": statistics.mean(ttft_times) if ttft_times else 0,
                "median": statistics.median(ttft_times) if ttft_times else 0,
                "min": min(ttft_times) if ttft_times else 0,
                "max": max(ttft_times) if ttft_times else 0,
                "samples": len(ttft_times)
            },
            "throughput": {
                "calls_per_second": throughput,
                "total_calls": total_calls,
                "total_seconds": total_time
            },
            "goodput": {
                "bytes_per_second": goodput,
                "total_useful_bytes": total_useful_bytes
            },
            "audio_metrics": {
                "avg_chunks_sent": statistics.mean([c.audio_chunks_sent for c in completed]),
                "avg_chunks_received": statistics.mean([c.audio_chunks_received for c in completed]),
                "total_calls_completed": len(completed)
            }
        }