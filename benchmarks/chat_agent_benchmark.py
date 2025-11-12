import asyncio
import json
import logging
import statistics
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import matplotlib.pyplot as plt
import pandas as pd
import websockets

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedChatAgentBenchmark:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        ws_base_url: str = "ws://localhost:8000",
    ):
        self.base_url = base_url.rstrip("/")
        self.ws_base_url = ws_base_url.rstrip("/")
        self.session_ids = []
        self.results = []
        self.timeout_count = 0
        self.error_count = 0

    async def initialize_session(self) -> str:
        """Initialize a chat session with better error handling"""
        endpoints = [
            f"{self.base_url}/messenger/chat/init",
            "http://localhost:40331/messenger/chat/init",
        ]

        for endpoint in endpoints:
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as session:
                    async with session.post(
                        endpoint,
                        json={
                            "owner_id": "+12345952496",
                            "agent_id": "AGENT_ID",
                            "user_id": "USER_ID",
                        },
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data["status"] == "success":
                                session_id = data["session_id"]
                                self.session_ids.append(session_id)
                                logger.info(f"Session initialized: {session_id}")
                                return session_id
            except Exception as e:
                logger.warning(f"Endpoint {endpoint} failed: {e}")
                continue

        raise Exception("All session initialization endpoints failed")

    async def send_message_and_measure(
        self, session_id: str, message: str, test_id: str
    ) -> Dict[str, Any]:
        """Enhanced message sending with better timeout handling and Arabic text support"""
        metrics = {
            "test_id": test_id,
            "message": message,
            "session_id": session_id,
            "ttft": None,
            "tpot": [],
            "itl": [],
            "total_tokens": 0,
            "total_time": 0,
            "goodput": 0,
            "start_time": time.time(),
            "first_token_time": None,
            "last_token_time": None,
            "status": "success",
            "error_message": None,
            "received_complete": False,
        }

        try:
            # WebSocket connection with timeout
            ws_url = f"{self.ws_base_url}/messenger/ws/{session_id}"
            async with websockets.connect(
                ws_url, ping_timeout=20, close_timeout=10
            ) as websocket:

                # Wait for connection confirmation with timeout
                try:
                    connected_msg = await asyncio.wait_for(
                        websocket.recv(), timeout=10.0
                    )
                    logger.info(f"Connected: {connected_msg}")
                except asyncio.TimeoutError:
                    metrics["status"] = "error"
                    metrics["error_message"] = "Connection timeout"
                    return metrics

                # Send message
                await websocket.send(message)
                token_count = 0
                last_token_time = None
                response_start_time = time.time()

                # Increased timeout for response processing
                while time.time() - response_start_time < 60:  # 60 second total timeout
                    try:
                        # Reduced timeout for individual message reception
                        response = await asyncio.wait_for(
                            websocket.recv(), timeout=15.0
                        )
                        data = json.loads(response)

                        current_time = time.time()

                        if data["type"] == "typing":
                            logger.info("AI is thinking...")

                        elif data["type"] == "chunk":
                            token_count += 1

                            # Measure TTFT
                            if metrics["first_token_time"] is None:
                                metrics["first_token_time"] = current_time
                                metrics["ttft"] = current_time - metrics["start_time"]
                                logger.info(
                                    f"TTFT: {metrics['ttft']:.3f}s - Token: '{data['content']}'"
                                )

                            # Measure ITL
                            if last_token_time is not None:
                                token_latency = current_time - last_token_time
                                metrics["itl"].append(token_latency)

                            last_token_time = current_time
                            logger.info(f"Token {token_count}: '{data['content']}'")

                        elif data["type"] == "complete":
                            metrics["last_token_time"] = current_time
                            metrics["total_time"] = current_time - metrics["start_time"]
                            metrics["total_tokens"] = token_count
                            metrics["received_complete"] = True

                            # Calculate final metrics
                            if (
                                token_count > 0
                                and metrics["first_token_time"] is not None
                            ):
                                generation_time = (
                                    metrics["last_token_time"]
                                    - metrics["first_token_time"]
                                )
                                metrics["goodput"] = (
                                    token_count / metrics["total_time"]
                                    if metrics["total_time"] > 0
                                    else 0
                                )
                                metrics["final_tpot"] = (
                                    generation_time / token_count
                                    if generation_time > 0
                                    else 0
                                )

                            logger.info(f"Complete: {data['message']}")
                            break

                        elif data["type"] == "error":
                            metrics["status"] = "error"
                            metrics["error_message"] = data["message"]
                            logger.error(f"Server error: {data['message']}")
                            break

                        elif data["type"] == "timeout":
                            metrics["status"] = "timeout"
                            metrics["error_message"] = data["message"]
                            logger.warning(f"Session timeout: {data['message']}")
                            break

                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for individual message")
                        # Continue waiting for next message instead of breaking immediately
                        continue

                # Check if we timed out overall
                if not metrics["received_complete"] and metrics["status"] == "success":
                    if token_count > 0:
                        metrics["status"] = "partial"
                        metrics["error_message"] = "Response incomplete - timeout"
                        logger.warning("Partial response received before timeout")
                    else:
                        metrics["status"] = "timeout"
                        metrics["error_message"] = "No response received - timeout"
                        self.timeout_count += 1

        except asyncio.TimeoutError:
            metrics["status"] = "timeout"
            metrics["error_message"] = "WebSocket connection timeout"
            self.timeout_count += 1
            logger.error("WebSocket connection timeout")
        except Exception as e:
            metrics["status"] = "error"
            metrics["error_message"] = str(e)
            self.error_count += 1
            logger.error(f"Error during message exchange: {e}")

        # Calculate metrics even for partial responses
        if token_count > 0 and metrics["first_token_time"] is not None:
            final_time = metrics["last_token_time"] or time.time()
            metrics["total_time"] = final_time - metrics["start_time"]
            metrics["total_tokens"] = token_count

            if metrics["first_token_time"] is not None:
                generation_time = final_time - metrics["first_token_time"]
                metrics["goodput"] = (
                    token_count / metrics["total_time"]
                    if metrics["total_time"] > 0
                    else 0
                )
                metrics["final_tpot"] = (
                    generation_time / token_count if generation_time > 0 else 0
                )

            # ITL statistics
            if metrics["itl"]:
                metrics["avg_itl"] = statistics.mean(metrics["itl"])
                metrics["max_itl"] = max(metrics["itl"])
                metrics["min_itl"] = min(metrics["itl"])
                metrics["std_itl"] = (
                    statistics.stdev(metrics["itl"]) if len(metrics["itl"]) > 1 else 0
                )

        return metrics

    async def run_single_test(self, message: str, test_id: str) -> Dict[str, Any]:
        """Run a single test with comprehensive error handling"""
        try:
            session_id = await self.initialize_session()
            logger.info(f"Starting test {test_id} with session: {session_id}")

            metrics = await self.send_message_and_measure(session_id, message, test_id)

            # Add test metadata
            metrics["message_length"] = len(message)
            metrics["timestamp"] = datetime.now().isoformat()

            return metrics

        except Exception as e:
            logger.error(f"Test {test_id} failed: {e}")
            return {
                "test_id": test_id,
                "status": "error",
                "error_message": str(e),
                "message_length": len(message),
                "timestamp": datetime.now().isoformat(),
            }

    async def run_benchmark_suite(
        self, test_scenarios: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Run comprehensive benchmark with different scenarios"""
        all_results = []

        for scenario in test_scenarios:
            logger.info(f"\n=== Testing Scenario: {scenario['name']} ===")

            if scenario["type"] == "concurrent":
                # Run concurrent tests
                results = await self.run_concurrent_tests(
                    messages=scenario["messages"],
                    concurrent_users=scenario.get("concurrent_users", 3),
                )
            elif scenario["type"] == "sequential":
                # Run sequential tests
                results = []
                for i, message in enumerate(scenario["messages"]):
                    test_id = f"{scenario['name']}_seq_{i+1}"
                    result = await self.run_single_test(message, test_id)
                    results.append(result)
                    # Wait between sequential tests
                    await asyncio.sleep(1)

            all_results.extend(results)

            # Wait between scenarios
            await asyncio.sleep(2)

        return all_results

    async def run_concurrent_tests(
        self, messages: List[str], concurrent_users: int = 3
    ) -> List[Dict[str, Any]]:
        """Run multiple tests concurrently"""
        logger.info(
            f"Running {len(messages)} tests with {concurrent_users} concurrent users"
        )

        batches = [
            messages[i : i + concurrent_users]
            for i in range(0, len(messages), concurrent_users)
        ]
        all_results = []

        for batch_num, batch in enumerate(batches):
            logger.info(f"Running batch {batch_num + 1}/{len(batches)}")

            tasks = []
            for i, message in enumerate(batch):
                test_id = f"batch{batch_num+1}_user{i+1}"
                task = self.run_single_test(message, test_id)
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Test resulted in exception: {result}")
                    continue
                if result is not None:
                    all_results.append(result)

            # Wait between batches
            if batch_num < len(batches) - 1:
                await asyncio.sleep(3)  # Increased wait time

        return all_results

    def analyze_performance_issues(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze performance issues and provide recommendations"""
        successful_tests = [r for r in results if r.get("status") == "success"]
        partial_tests = [r for r in results if r.get("status") == "partial"]
        timeout_tests = [r for r in results if r.get("status") == "timeout"]
        error_tests = [r for r in results if r.get("status") == "error"]

        analysis = {
            "summary": {
                "total_tests": len(results),
                "successful": len(successful_tests),
                "partial": len(partial_tests),
                "timeout": len(timeout_tests),
                "error": len(error_tests),
                "success_rate": len(successful_tests) / len(results) * 100,
            },
            "issues": [],
            "recommendations": [],
        }

        # Analyze timeouts
        if timeout_tests:
            analysis["issues"].append(
                f"High timeout rate: {len(timeout_tests)}/{len(results)} tests timed out"
            )
            analysis["recommendations"].extend(
                [
                    "Increase server timeout settings",
                    "Check backend service health and resource usage",
                    "Optimize LLM inference performance",
                    "Consider implementing request queuing for high load",
                ]
            )

        # Analyze TTFT performance
        if successful_tests:
            ttft_values = [r["ttft"] for r in successful_tests if r.get("ttft")]
            if ttft_values:
                avg_ttft = statistics.mean(ttft_values)
                if avg_ttft > 1.0:
                    analysis["issues"].append(f"High TTFT: {avg_ttft:.2f}s average")
                    analysis["recommendations"].append(
                        "Optimize first token generation latency"
                    )

        # Analyze token generation consistency
        if successful_tests:
            itl_values = []
            for r in successful_tests:
                itl_values.extend(r.get("itl", []))

            if itl_values and len(itl_values) > 1:
                itl_std = statistics.stdev(itl_values)
                if itl_std > 0.5:  # High variance in token latency
                    analysis["issues"].append("High variance in inter-token latency")
                    analysis["recommendations"].append(
                        "Investigate token generation consistency"
                    )

        return analysis

    def generate_comprehensive_report(self, results: List[Dict[str, Any]]):
        """Generate detailed benchmark report with analysis"""
        successful_tests = [r for r in results if r.get("status") == "success"]
        partial_tests = [r for r in results if r.get("status") == "partial"]

        print("\n" + "=" * 80)
        print("ENHANCED CHAT AGENT BENCHMARK REPORT")
        print("=" * 80)

        # Performance analysis
        analysis = self.analyze_performance_issues(results)

        print(f"\nTEST SUMMARY:")
        print(f"  Total Tests: {analysis['summary']['total_tests']}")
        print(f"  Successful: {analysis['summary']['successful']}")
        print(f"  Partial: {analysis['summary']['partial']}")
        print(f"  Timeout: {analysis['summary']['timeout']}")
        print(f"  Error: {analysis['summary']['error']}")
        print(f"  Success Rate: {analysis['summary']['success_rate']:.1f}%")

        if successful_tests:
            print(f"\nPERFORMANCE METRICS (Successful Tests):")

            # TTFT
            ttft_values = [r["ttft"] for r in successful_tests if r.get("ttft")]
            if ttft_values:
                print(f"  TTFT (Time To First Token):")
                print(f"    Average: {statistics.mean(ttft_values):.3f}s")
                print(f"    Median:  {statistics.median(ttft_values):.3f}s")
                print(f"    Range:   {min(ttft_values):.3f}s - {max(ttft_values):.3f}s")
                print(
                    f"    Std Dev: {statistics.stdev(ttft_values) if len(ttft_values) > 1 else 0:.3f}s"
                )

            # TPOT
            tpot_values = [
                r.get("final_tpot", 0) for r in successful_tests if r.get("final_tpot")
            ]
            if tpot_values:
                print(f"  TPOT (Time Per Output Token):")
                print(f"    Average: {statistics.mean(tpot_values):.3f}s/token")
                print(f"    Median:  {statistics.median(tpot_values):.3f}s/token")
                print(f"    Range:   {min(tpot_values):.3f}s - {max(tpot_values):.3f}s")

            # Goodput
            goodput_values = [
                r["goodput"] for r in successful_tests if r.get("goodput")
            ]
            if goodput_values:
                print(f"  Goodput (Tokens per Second):")
                print(f"    Average: {statistics.mean(goodput_values):.1f} tokens/s")
                print(f"    Median:  {statistics.median(goodput_values):.1f} tokens/s")
                print(
                    f"    Range:   {min(goodput_values):.1f} - {max(goodput_values):.1f} tokens/s"
                )

            # ITL
            all_itl = []
            for r in successful_tests:
                all_itl.extend(r.get("itl", []))

            if all_itl:
                print(f"  ITL (Inter-Token Latency):")
                print(f"    Average: {statistics.mean(all_itl):.3f}s")
                print(f"    Median:  {statistics.median(all_itl):.3f}s")
                print(f"    Range:   {min(all_itl):.3f}s - {max(all_itl):.3f}s")
                print(
                    f"    Std Dev: {statistics.stdev(all_itl) if len(all_itl) > 1 else 0:.3f}s"
                )

        # Issues and Recommendations
        if analysis["issues"]:
            print(f"\nIDENTIFIED ISSUES:")
            for issue in analysis["issues"]:
                print(f"  ⚠️  {issue}")

        if analysis["recommendations"]:
            print(f"\nRECOMMENDATIONS:")
            for rec in analysis["recommendations"]:
                print(f"  ✅ {rec}")

        # Save detailed results
        self.save_detailed_results(results)

    def save_detailed_results(self, results: List[Dict[str, Any]]):
        """Save comprehensive results to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_benchmark_results_{timestamp}.csv"

        df_data = []
        for result in results:
            row = {
                "test_id": result.get("test_id", ""),
                "status": result.get("status", "unknown"),
                "message_length": result.get("message_length", 0),
                "ttft": result.get("ttft"),
                "final_tpot": result.get("final_tpot"),
                "goodput": result.get("goodput"),
                "total_tokens": result.get("total_tokens", 0),
                "total_time": result.get("total_time", 0),
                "avg_itl": statistics.mean(result["itl"]) if result.get("itl") else 0,
                "error_message": result.get("error_message", ""),
                "timestamp": result.get("timestamp", ""),
            }
            df_data.append(row)

        if df_data:
            df = pd.DataFrame(df_data)
            df.to_csv(filename, index=False)
            print(f"\nDetailed results saved to: {filename}")

    async def cleanup(self):
        """Clean up sessions"""
        for session_id in self.session_ids:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.delete(
                        f"{self.base_url}/messenger/chat/{session_id}"
                    ) as response:
                        if response.status == 200:
                            logger.info(f"Cleaned up session: {session_id}")
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")


async def main():
    """Main benchmarking function with comprehensive test scenarios"""
    benchmark = EnhancedChatAgentBenchmark()

    # Define comprehensive test scenarios
    test_scenarios = [
        {
            "name": "Simple Greetings",
            "type": "concurrent",
            "concurrent_users": 2,
            "messages": [
                "Hello, how are you?",
                "Hi there!",
                "Good morning",
                "Hello, can you help me?",
            ],
        },
        {
            "name": "Factual Questions",
            "type": "concurrent",
            "concurrent_users": 2,
            "messages": [
                "What is the capital of France?",
                "Explain quantum computing briefly.",
                "What is photosynthesis?",
                "How does the internet work?",
            ],
        },
        {
            "name": "Complex Reasoning",
            "type": "sequential",  # Run sequentially to avoid overload
            "messages": [
                "Explain the theory of relativity in simple terms.",
                "What are the ethical implications of artificial intelligence?",
                "How can we solve climate change?",
                "What is the future of renewable energy?",
            ],
        },
    ]

    try:
        logger.info("Starting Enhanced Chat Agent Benchmark...")

        # Run comprehensive benchmark suite
        results = await benchmark.run_benchmark_suite(test_scenarios)

        # Generate comprehensive report
        benchmark.generate_comprehensive_report(results)

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
    finally:
        await benchmark.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
