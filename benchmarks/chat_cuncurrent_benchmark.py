import asyncio
import json
import logging
import statistics
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
import websockets

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HighConcurrencyBenchmark:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        ws_base_url: str = "ws://localhost:8000",
        max_concurrent: int = 10,
    ):
        self.base_url = base_url.rstrip("/")
        self.ws_base_url = ws_base_url.rstrip("/")
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.results = []
        self.session_ids = []

    async def initialize_session(self) -> str:
        """Initialize session with semaphore for concurrency control"""
        async with self.semaphore:
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as session:
                    async with session.post(
                        f"{self.base_url}/messenger/chat/init",
                        json={
                            "owner_id": "+12345952496",
                            "agent_id": "AGENT_ID",
                            "user_id": f"USER_{int(time.time())}_{uuid.uuid4().hex[:8]}",
                        },
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            session_id = data["session_id"]
                            self.session_ids.append(session_id)
                            return session_id
                        else:
                            raise Exception(
                                f"HTTP {response.status}: {await response.text()}"
                            )
            except Exception as e:
                logging.error(f"Session init failed: {e}")
                raise

    async def send_single_message(
        self, session_id: str, message: str, test_id: str
    ) -> Dict[str, Any]:
        """Send a single message and measure performance"""
        metrics = {
            "test_id": test_id,
            "session_id": session_id,
            "message": message,
            "ttft": None,
            "total_tokens": 0,
            "total_time": 0,
            "status": "success",
            "error_message": None,
        }

        try:
            ws_url = f"{self.ws_base_url}/messenger/ws/{session_id}"
            async with websockets.connect(
                ws_url, ping_timeout=20, close_timeout=10
            ) as websocket:

                # Wait for connection confirmation
                await asyncio.wait_for(websocket.recv(), timeout=10.0)

                start_time = time.time()
                await websocket.send(message)

                token_count = 0
                first_token_time = None

                while True:
                    try:
                        response = await asyncio.wait_for(
                            websocket.recv(), timeout=30.0
                        )
                        data = json.loads(response)
                        current_time = time.time()

                        if data["type"] == "chunk":
                            token_count += 1
                            if first_token_time is None:
                                first_token_time = current_time
                                metrics["ttft"] = current_time - start_time

                        elif data["type"] == "complete":
                            metrics["total_time"] = current_time - start_time
                            metrics["total_tokens"] = token_count
                            break

                        elif data["type"] in ["error", "timeout"]:
                            metrics["status"] = "error"
                            metrics["error_message"] = data.get(
                                "message", "Unknown error"
                            )
                            break

                    except asyncio.TimeoutError:
                        metrics["status"] = "timeout"
                        metrics["error_message"] = "Response timeout"
                        break

        except Exception as e:
            metrics["status"] = "error"
            metrics["error_message"] = str(e)

        return metrics

    async def run_conversation_test(
        self, message: str, test_id: str, conversation_length: int = 1
    ):
        """Run a test with a conversation (single message for stress testing)"""
        try:
            session_id = await self.initialize_session()
            results = []

            for i in range(conversation_length):
                result = await self.send_single_message(
                    session_id=session_id,
                    message=message,
                    test_id=f"{test_id}_turn_{i+1}",
                )
                results.append(result)

            return results[0]  # Return first turn result for stress testing

        except Exception as e:
            logger.error(f"Conversation test failed: {e}")
            return {"test_id": test_id, "status": "error", "error_message": str(e)}

    async def stress_test_concurrent_users(self, num_users: int, message: str):
        """Test with high number of concurrent users"""
        logging.info(f"Starting stress test with {num_users} concurrent users")

        tasks = []
        for i in range(num_users):
            task = self.run_conversation_test(
                message=message,
                test_id=f"stress_user_{i+1}",
                conversation_length=1,  # Single message for stress test
            )
            tasks.append(task)

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = [
            r
            for r in results
            if not isinstance(r, Exception) and r.get("status") == "success"
        ]
        failed = [
            r
            for r in results
            if isinstance(r, Exception) or r.get("status") != "success"
        ]

        logging.info(
            f"Stress test completed: {len(successful)} successful, {len(failed)} failed"
        )
        return successful, failed

    async def run_scalability_test(self, max_users: int = 20, step: int = 2):
        """Test scalability by gradually increasing concurrent users"""
        scalability_results = []

        for num_users in range(1, max_users + 1, step):
            logger.info(f"Testing with {num_users} concurrent users...")

            start_time = time.time()
            successful, failed = await self.stress_test_concurrent_users(
                num_users=num_users, message="Hello, how are you?"
            )
            end_time = time.time()

            metrics = {
                "concurrent_users": num_users,
                "successful_requests": len(successful),
                "failed_requests": len(failed),
                "success_rate": len(successful) / num_users * 100,
                "total_time": end_time - start_time,
                "requests_per_second": (
                    len(successful) / (end_time - start_time)
                    if (end_time - start_time) > 0
                    else 0
                ),
            }

            if successful:
                ttft_values = [r["ttft"] for r in successful if r.get("ttft")]
                if ttft_values:
                    metrics["avg_ttft"] = statistics.mean(ttft_values)
                    metrics["min_ttft"] = min(ttft_values)
                    metrics["max_ttft"] = max(ttft_values)
                    metrics["std_ttft"] = (
                        statistics.stdev(ttft_values) if len(ttft_values) > 1 else 0
                    )
                else:
                    metrics["avg_ttft"] = 0
                    metrics["min_ttft"] = 0
                    metrics["max_ttft"] = 0
                    metrics["std_ttft"] = 0

            scalability_results.append(metrics)
            logger.info(
                f"Concurrent Users: {num_users}, Success Rate: {metrics['success_rate']:.1f}%, Avg TTFT: {metrics.get('avg_ttft', 0):.3f}s"
            )

            # Wait between scalability steps
            if num_users < max_users:
                await asyncio.sleep(3)

        return scalability_results

    async def run_heavy_message_test(self, concurrent_users: int = 5):
        """Test with heavier messages to simulate real workload"""
        heavy_messages = [
            "Can you explain the concept of machine learning in detail?",
            "What are the main differences between supervised and unsupervised learning?",
            "Describe the transformer architecture and how it revolutionized NLP.",
            "How does attention mechanism work in neural networks?",
            "Explain the backpropagation algorithm step by step.",
        ]

        logger.info(f"Running heavy message test with {concurrent_users} users")

        tasks = []
        for i in range(concurrent_users):
            message = heavy_messages[i % len(heavy_messages)]
            task = self.run_conversation_test(
                message=message,
                test_id=f"heavy_user_{i+1}",
                conversation_length=1,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = [
            r
            for r in results
            if not isinstance(r, Exception) and r.get("status") == "success"
        ]

        return {
            "total_users": concurrent_users,
            "successful": len(successful),
            "success_rate": len(successful) / concurrent_users * 100,
            "avg_ttft": (
                statistics.mean([r["ttft"] for r in successful if r.get("ttft")])
                if successful
                else 0
            ),
        }

    async def cleanup(self):
        """Clean up all sessions"""
        cleanup_tasks = []
        for session_id in self.session_ids:
            try:
                async with aiohttp.ClientSession() as session:
                    task = session.delete(
                        f"{self.base_url}/messenger/chat/{session_id}"
                    )
                    cleanup_tasks.append(task)
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            logger.info(f"Cleaned up {len(cleanup_tasks)} sessions")


async def run_high_concurrency_test():
    benchmark = HighConcurrencyBenchmark(max_concurrent=10)

    try:
        print("🚀 Starting High Concurrency Benchmark")
        print("=" * 60)

        # Test 1: Scalability test
        print("\n📈 Running Scalability Test...")
        scalability_results = await benchmark.run_scalability_test(max_users=15, step=3)

        print("\n" + "=" * 60)
        print("SCALABILITY TEST RESULTS")
        print("=" * 60)
        print(
            f"{'Users':>6} {'Success':>8} {'RPS':>8} {'Avg TTFT':>10} {'Min TTFT':>10} {'Max TTFT':>10}"
        )
        print("-" * 60)

        for result in scalability_results:
            print(
                f"{result['concurrent_users']:6d} | "
                f"{result['success_rate']:7.1f}% | "
                f"{result['requests_per_second']:7.2f} | "
                f"{result.get('avg_ttft', 0):8.3f}s | "
                f"{result.get('min_ttft', 0):8.3f}s | "
                f"{result.get('max_ttft', 0):8.3f}s"
            )

        # Find the breaking point
        breaking_point = None
        for result in scalability_results:
            if result["success_rate"] < 80:  # Below 80% success rate
                breaking_point = result["concurrent_users"]
                break

        if breaking_point:
            print(f"\n🚨 SYSTEM BREAKING POINT: {breaking_point} concurrent users")
        else:
            print(
                f"\n✅ System handles up to {scalability_results[-1]['concurrent_users']} users successfully"
            )

        # Test 2: Heavy message test
        print("\n🔧 Running Heavy Message Test...")
        heavy_results = await benchmark.run_heavy_message_test(concurrent_users=5)
        print(f"Heavy Message Test Results:")
        print(f"  Users: {heavy_results['total_users']}")
        print(f"  Success Rate: {heavy_results['success_rate']:.1f}%")
        print(f"  Avg TTFT: {heavy_results['avg_ttft']:.3f}s")

        # Generate summary report
        print("\n" + "=" * 60)
        print("SUMMARY REPORT")
        print("=" * 60)

        total_tests = (
            sum(r["concurrent_users"] for r in scalability_results)
            + heavy_results["total_users"]
        )
        total_success = (
            sum(r["successful_requests"] for r in scalability_results)
            + heavy_results["successful"]
        )

        print(f"Total Tests Run: {total_tests}")
        print(f"Total Successful: {total_success}")
        print(f"Overall Success Rate: {total_success/total_tests*100:.1f}%")

        # Performance recommendations
        max_stable_users = (
            breaking_point - 3
            if breaking_point
            else scalability_results[-1]["concurrent_users"]
        )
        print(
            f"\n💡 RECOMMENDATION: Limit to {max_stable_users} concurrent users for stable performance"
        )

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
    finally:
        await benchmark.cleanup()


async def quick_concurrency_test():
    """Quick test to check basic concurrency"""
    benchmark = HighConcurrencyBenchmark(max_concurrent=5)

    try:
        print("🚀 Quick Concurrency Test (5 users)")
        successful, failed = await benchmark.stress_test_concurrent_users(
            num_users=5, message="Hello, quick test!"
        )

        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")

        if successful:
            ttft_values = [r["ttft"] for r in successful if r.get("ttft")]
            if ttft_values:
                print(
                    f"📊 TTFT - Avg: {statistics.mean(ttft_values):.3f}s, "
                    f"Min: {min(ttft_values):.3f}s, Max: {max(ttft_values):.3f}s"
                )

    finally:
        await benchmark.cleanup()


if __name__ == "__main__":
    # Run quick test first to verify everything works
    # asyncio.run(quick_concurrency_test())

    # Then run full benchmark
    asyncio.run(run_high_concurrency_test())
