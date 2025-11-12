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


class MemoryHistoryBenchmark:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        ws_base_url: str = "ws://localhost:8000",
    ):
        self.base_url = base_url.rstrip("/")
        self.ws_base_url = ws_base_url.rstrip("/")
        self.results = []
        self.session_ids = []  # Add this missing attribute

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

    async def test_conversation_memory(
        self, session_id: str, conversation_flow: List[Dict], test_id: str
    ):
        """Test memory across a long conversation"""
        metrics = {
            "test_id": test_id,
            "session_id": session_id,
            "conversation_length": len(conversation_flow),
            "turn_metrics": [],
            "memory_success": False,
            "final_context_accuracy": 0,
        }

        try:
            ws_url = f"{self.ws_base_url}/messenger/ws/{session_id}"
            async with websockets.connect(ws_url) as websocket:
                # Wait for connection
                await websocket.recv()

                conversation_context = []

                for turn_num, turn in enumerate(conversation_flow):
                    turn_metrics = {
                        "turn_number": turn_num + 1,
                        "user_message": turn["user"],
                        "expected_context": turn.get("context", []),
                        "ttft": None,
                        "response_time": None,
                        "response_quality": 0,
                        "memory_retention": 0,
                    }

                    start_time = time.time()
                    await websocket.send(turn["user"])

                    response_chunks = []
                    first_token = True

                    while True:
                        response = await asyncio.wait_for(
                            websocket.recv(), timeout=30.0
                        )
                        data = json.loads(response)

                        if data["type"] == "chunk":
                            if first_token:
                                turn_metrics["ttft"] = time.time() - start_time
                                first_token = False
                            response_chunks.append(data["content"])

                        elif data["type"] == "complete":
                            turn_metrics["response_time"] = time.time() - start_time
                            full_response = "".join(response_chunks)

                            # Analyze response quality and memory retention
                            quality_score = self.analyze_response_quality(
                                user_message=turn["user"],
                                ai_response=full_response,
                                expected_context=turn.get("context", []),
                                previous_context=conversation_context,
                            )

                            turn_metrics["response_quality"] = quality_score["overall"]
                            turn_metrics["memory_retention"] = quality_score[
                                "memory_retention"
                            ]

                            # Update conversation context
                            conversation_context.append(
                                {"user": turn["user"], "assistant": full_response}
                            )
                            break

                    metrics["turn_metrics"].append(turn_metrics)
                    logging.info(
                        f"Turn {turn_num + 1}: TTFT={turn_metrics['ttft']:.3f}s, "
                        f"Memory={turn_metrics['memory_retention']:.1f}%"
                    )

                # Final memory test
                final_memory_score = await self.test_final_memory(
                    websocket, conversation_context
                )
                metrics["final_context_accuracy"] = final_memory_score
                metrics["memory_success"] = (
                    final_memory_score > 0.7
                )  # 70% accuracy threshold

        except Exception as e:
            logging.error(f"Memory test failed: {e}")
            metrics["error"] = str(e)

        return metrics

    def analyze_response_quality(
        self,
        user_message: str,
        ai_response: str,
        expected_context: List[str],
        previous_context: List[Dict],
    ) -> Dict[str, float]:
        """Analyze if the response maintains conversation context"""
        quality_metrics = {
            "relevance": 1.0,
            "memory_retention": 0.0,
            "consistency": 1.0,
            "overall": 0.0,
        }

        # Check for context retention
        context_keywords = []
        for ctx in previous_context[-3:]:  # Last 3 turns
            context_keywords.extend(
                self.extract_keywords(ctx["user"] + " " + ctx["assistant"])
            )

        # Calculate memory retention score
        memory_hits = 0
        for keyword in context_keywords[-10:]:  # Check recent keywords
            if keyword.lower() in ai_response.lower():
                memory_hits += 1

        if context_keywords:
            quality_metrics["memory_retention"] = memory_hits / len(
                context_keywords[-10:]
            )

        # Check expected context
        expected_hits = 0
        for expected in expected_context:
            if expected.lower() in ai_response.lower():
                expected_hits += 1

        if expected_context:
            quality_metrics["relevance"] = expected_hits / len(expected_context)

        # Overall score (weighted average)
        quality_metrics["overall"] = (
            quality_metrics["relevance"] * 0.4
            + quality_metrics["memory_retention"] * 0.4
            + quality_metrics["consistency"] * 0.2
        )

        return quality_metrics

    async def test_final_memory(
        self, websocket, conversation_context: List[Dict]
    ) -> float:
        """Test if the model remembers the entire conversation"""
        if len(conversation_context) < 2:
            return 1.0  # Not enough context to test

        # Ask about early conversation details
        early_topic = conversation_context[0]["user"]
        memory_test_message = (
            f"What was the first thing I asked you about? Please be specific."
        )

        await websocket.send(memory_test_message)
        response_chunks = []

        while True:
            response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            data = json.loads(response)

            if data["type"] == "chunk":
                response_chunks.append(data["content"])
            elif data["type"] == "complete":
                final_response = "".join(response_chunks)

                # Check if response contains early conversation context
                early_keywords = self.extract_keywords(early_topic)
                memory_hits = sum(
                    1
                    for keyword in early_keywords
                    if keyword.lower() in final_response.lower()
                )

                return memory_hits / len(early_keywords) if early_keywords else 0.0

    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction - you can enhance this
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "but"}
        words = text.lower().split()
        return [word for word in words if len(word) > 3 and word not in stop_words][
            :5
        ]  # Top 5 keywords


def generate_memory_report(results: List[Dict]):
    """Generate detailed memory performance report"""
    print("\n" + "=" * 70)
    print("CONVERSATION MEMORY BENCHMARK REPORT")
    print("=" * 70)

    successful_tests = [r for r in results if r.get("memory_success")]

    print(f"\nOverall Memory Performance:")
    print(f"  Total Tests: {len(results)}")
    print(f"  Successful Memory: {len(successful_tests)}")
    print(f"  Memory Success Rate: {len(successful_tests)/len(results)*100:.1f}%")

    if successful_tests:
        # Analyze TTFT progression in long conversations
        all_turn_metrics = []
        for result in successful_tests:
            all_turn_metrics.extend(result["turn_metrics"])

        # Group by turn number
        turns_data = {}
        for metric in all_turn_metrics:
            turn_num = metric["turn_number"]
            if turn_num not in turns_data:
                turns_data[turn_num] = []
            turns_data[turn_num].append(metric)

        print(f"\nTTFT Progression by Conversation Turn:")
        for turn_num in sorted(turns_data.keys()):
            ttft_values = [m["ttft"] for m in turns_data[turn_num] if m["ttft"]]
            if ttft_values:
                avg_ttft = statistics.mean(ttft_values)
                print(f"  Turn {turn_num}: {avg_ttft:.3f}s")

        # Memory retention analysis
        memory_scores = [r["final_context_accuracy"] for r in successful_tests]
        print(f"\nMemory Accuracy:")
        print(f"  Average: {statistics.mean(memory_scores):.1%}")
        print(f"  Range: {min(memory_scores):.1%} - {max(memory_scores):.1%}")


async def run_comprehensive_memory_tests():
    benchmark = MemoryHistoryBenchmark()

    # Define different conversation scenarios
    test_scenarios = [
        {
            "name": "Simple Q&A Memory",
            "conversation": [
                {
                    "user": "My name is John and I love pizza.",
                    "context": ["John", "pizza"],
                },
                {
                    "user": "What's my name and favorite food?",
                    "context": ["John", "pizza"],
                },
                {
                    "user": "Can you remind me what I told you about myself?",
                    "context": ["John", "pizza"],
                },
            ],
        },
        {
            "name": "Complex Story Memory",
            "conversation": [
                {
                    "user": "I'm planning a trip to Japan next month. I want to visit Tokyo and Kyoto.",
                    "context": ["Japan", "Tokyo", "Kyoto", "trip"],
                },
                {
                    "user": "What cities should I visit in Japan?",
                    "context": ["Tokyo", "Kyoto"],
                },
                {
                    "user": "Tell me more about Japanese culture for my trip.",
                    "context": ["Japan", "culture"],
                },
                {
                    "user": "What was my original travel plan?",
                    "context": ["Japan", "Tokyo", "Kyoto"],
                },
            ],
        },
        {
            "name": "Technical Discussion",
            "conversation": [
                {
                    "user": "I'm working on a Python project using FastAPI and MongoDB.",
                    "context": ["Python", "FastAPI", "MongoDB"],
                },
                {
                    "user": "How can I optimize database queries in this setup?",
                    "context": ["database", "queries", "MongoDB"],
                },
                {
                    "user": "What framework and database was I using again?",
                    "context": ["FastAPI", "MongoDB"],
                },
                {
                    "user": "Can you summarize my tech stack?",
                    "context": ["Python", "FastAPI", "MongoDB"],
                },
            ],
        },
    ]

    all_results = []

    for scenario in test_scenarios:
        logging.info(f"\n🧠 Testing Memory: {scenario['name']}")

        # Initialize session for this conversation
        session_id = await benchmark.initialize_session()

        # Run memory test
        result = await benchmark.test_conversation_memory(
            session_id=session_id,
            conversation_flow=scenario["conversation"],
            test_id=scenario["name"],
        )

        all_results.append(result)

        # Print conversation summary
        print(f"\n{scenario['name']} Results:")
        print(f"  Memory Success: {result['memory_success']}")
        print(f"  Final Accuracy: {result['final_context_accuracy']:.1%}")
        print(f"  Conversation Length: {result['conversation_length']} turns")

        # Show TTFT progression
        ttft_values = [turn["ttft"] for turn in result["turn_metrics"]]
        if ttft_values:
            print(f"  TTFT Range: {min(ttft_values):.3f}s - {max(ttft_values):.3f}s")
            print(f"  TTFT Trend: {'↑' if ttft_values[-1] > ttft_values[0] else '↓'}")

    # Generate comprehensive memory report
    generate_memory_report(all_results)


if __name__ == "__main__":
    asyncio.run(run_comprehensive_memory_tests())
