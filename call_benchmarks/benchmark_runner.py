import asyncio
import json
import os
import time
from typing import List

from benchmark_config import BenchmarkConfig
from metrics_collector import MetricsCollector
from user_simulator import UserSimulator


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.audio_files = self._discover_audio_files()

    def _discover_audio_files(self) -> List[str]:
        """Discover all WAV files in the audio directory"""
        audio_files = []
        if os.path.exists(self.config.audio_files_dir):
            for file in os.listdir(self.config.audio_files_dir):
                if file.lower().endswith(".wav"):
                    audio_files.append(os.path.join(self.config.audio_files_dir, file))

        if not audio_files:
            print(f"Warning: No WAV files found in {self.config.audio_files_dir}")
            # Create a dummy list for simulation
            audio_files = ["./test_audio/dummy.wav"] * 5

        return audio_files

    async def run_benchmark(self):
        """Run the complete benchmark"""
        print("Starting benchmark...")
        print(
            f"Configuration: {self.config.max_concurrent_users} users, {self.config.test_duration_seconds}s duration"
        )

        # Warm-up phase
        print("Starting warm-up phase...")
        await self._warm_up()

        # Main test phase
        print("Starting main test phase...")
        start_time = time.time()

        tasks = []
        for user_batch in range(
            0, self.config.max_concurrent_users, self.config.ramp_up_users
        ):
            # Start a batch of users
            batch_tasks = []
            for user_id in range(
                user_batch,
                min(
                    user_batch + self.config.ramp_up_users,
                    self.config.max_concurrent_users,
                ),
            ):
                simulator = UserSimulator(
                    user_id=user_id,
                    base_url="http://localhost:8000",  # Adjust to your orchestrator URL
                    audio_files=self.audio_files,
                    metrics_collector=self.metrics_collector,
                )
                task = asyncio.create_task(self._run_user_simulation(simulator))
                batch_tasks.append(task)

            tasks.extend(batch_tasks)
            print(f"Started {len(batch_tasks)} users. Total active: {len(tasks)}")

            # Wait before starting next batch
            if (
                user_batch + self.config.ramp_up_users
                < self.config.max_concurrent_users
            ):
                await asyncio.sleep(self.config.ramp_up_delay)

        # Let the test run for the specified duration
        elapsed = 0
        while elapsed < self.config.test_duration_seconds and tasks:
            await asyncio.sleep(5)
            elapsed = time.time() - start_time

            # Print progress
            completed = len(self.metrics_collector.completed_calls)
            active = len(self.metrics_collector.active_calls)
            print(
                f"Progress: {elapsed:.1f}s / {self.config.test_duration_seconds}s - "
                f"Active: {active}, Completed: {completed}"
            )

        # Cancel remaining tasks
        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate and display results
        self._display_results()

    async def _warm_up(self):
        """Warm up the system with a few calls"""
        warmup_tasks = []
        for i in range(2):  # Start with 2 warm-up calls
            simulator = UserSimulator(
                user_id=i,
                base_url="http://localhost:8000",
                audio_files=self.audio_files,
                metrics_collector=self.metrics_collector,
            )
            task = asyncio.create_task(simulator.simulate_call())
            warmup_tasks.append(task)

        # Let warm-up run for a short time
        await asyncio.sleep(self.config.warm_up_seconds)
        for task in warmup_tasks:
            task.cancel()

        if warmup_tasks:
            await asyncio.gather(*warmup_tasks, return_exceptions=True)

    async def _run_user_simulation(self, simulator: UserSimulator):
        """Run a single user simulation with error handling"""
        try:
            await simulator.simulate_call()
        except Exception as e:
            print(f"User {simulator.user_id} simulation failed: {e}")

    def _display_results(self):
        """Display benchmark results"""
        results = self.metrics_collector.calculate_metrics()

        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)

        print(
            f"\nThroughput: {results['throughput']['calls_per_second']:.2f} calls/second"
        )
        print(f"Total Calls: {results['throughput']['total_calls']}")
        print(f"Total Duration: {results['throughput']['total_seconds']:.2f} seconds")

        print(f"\nTTFT (Time To First Token):")
        ttft = results["TTFT"]
        print(f"  Mean: {ttft['mean']:.3f}s")
        print(f"  Median: {ttft['median']:.3f}s")
        print(f"  Min: {ttft['min']:.3f}s")
        print(f"  Max: {ttft['max']:.3f}s")
        print(f"  Samples: {ttft['samples']}")

        print(f"\nITL (Input Latency):")
        itl = results["ITL"]
        print(f"  Mean: {itl['mean']:.3f}s")
        print(f"  Median: {itl['median']:.3f}s")
        print(f"  Min: {itl['min']:.3f}s")
        print(f"  Max: {itl['max']:.3f}s")
        print(f"  Samples: {itl['samples']}")

        print(f"\nGoodput: {results['goodput']['bytes_per_second']:.2f} bytes/second")

        print(f"\nAudio Metrics:")
        audio = results["audio_metrics"]
        print(f"  Average chunks sent: {audio['avg_chunks_sent']:.1f}")
        print(f"  Average chunks received: {audio['avg_chunks_received']:.1f}")

        # Save detailed results to file
        with open("benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nDetailed results saved to benchmark_results.json")


async def main():
    config = BenchmarkConfig(
        max_concurrent_users=10,  # Start with 10, you can increase this
        ramp_up_users=2,
        ramp_up_delay=1.0,
        test_duration_seconds=300,  # 5 minutes
        warm_up_seconds=30,
        audio_files_dir="/home/ubuntu/borhan/whole_pipeline/vexu/test_voices",
    )

    runner = BenchmarkRunner(config)
    await runner.run_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
