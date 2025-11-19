import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
import websockets


@dataclass
class MemoryMetrics:
    test_id: str
    session_id: str
    conversation_length: int
    turn_metrics: List[Dict]
    memory_success: bool
    final_context_accuracy: float
    memory_usage_trend: List[float]
    average_ttft: float
    average_memory_retention: float


class EnhancedMemoryHistoryBenchmark:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        ws_base_url: str = "ws://localhost:8000",
        max_conversation_turns: int = 50,
    ):
        self.base_url = base_url.rstrip("/")
        self.ws_base_url = ws_base_url.rstrip("/")
        self.max_conversation_turns = max_conversation_turns
        self.results = []
        self.session_ids = []
        self.memory_threshold = 0.7  # 70% memory retention threshold

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def initialize_session(self) -> str:
        """Initialize a chat session with retry logic"""
        endpoints = [
            f"{self.base_url}/messenger/chat/init",
            "http://localhost:40331/messenger/chat/init",
        ]

        for attempt in range(3):
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
                                    self.logger.info(
                                        f"Session initialized: {session_id}"
                                    )
                                    return session_id
                except Exception as e:
                    self.logger.warning(
                        f"Endpoint {endpoint} failed (attempt {attempt+1}): {e}"
                    )
                    continue

            await asyncio.sleep(1)  # Wait before retry

        raise Exception("All session initialization endpoints failed after 3 attempts")

    async def run_comprehensive_memory_test(
        self, test_scenarios: List[Dict], num_iterations: int = 3
    ) -> Dict[str, Any]:
        """Run multiple memory test scenarios"""
        comprehensive_results = {
            "test_timestamp": time.time(),
            "scenario_results": [],
            "aggregate_metrics": {},
            "memory_policy_recommendations": [],
        }

        for scenario in test_scenarios:
            scenario_results = []

            for iteration in range(num_iterations):
                self.logger.info(
                    f"Running scenario '{scenario['name']}' iteration {iteration+1}"
                )

                try:
                    session_id = await self.initialize_session()
                    metrics = await self.test_conversation_memory(
                        session_id=session_id,
                        conversation_flow=scenario["conversation_flow"],
                        test_id=f"{scenario['name']}_iter{iteration+1}",
                    )
                    scenario_results.append(metrics)

                    # Clean up session
                    await self.cleanup_session(session_id)

                except Exception as e:
                    self.logger.error(f"Scenario {scenario['name']} failed: {e}")
                    continue

            # Analyze scenario results
            if scenario_results:
                scenario_analysis = self.analyze_scenario_results(scenario_results)
                comprehensive_results["scenario_results"].append(scenario_analysis)

        # Generate overall recommendations
        comprehensive_results["memory_policy_recommendations"] = (
            self.generate_memory_policy_recommendations(
                comprehensive_results["scenario_results"]
            )
        )

        return comprehensive_results

    async def test_conversation_memory(
        self, session_id: str, conversation_flow: List[Dict], test_id: str
    ) -> Dict[str, Any]:
        """Enhanced memory testing with additional metrics"""
        metrics = {
            "test_id": test_id,
            "session_id": session_id,
            "conversation_length": len(conversation_flow),
            "turn_metrics": [],
            "memory_success": False,
            "final_context_accuracy": 0,
            "memory_usage_trend": [],
            "context_degradation_rate": 0,
            "peak_memory_retention": 0,
        }

        try:
            ws_url = f"{self.ws_base_url}/messenger/ws/{session_id}"
            async with websockets.connect(ws_url) as websocket:
                await websocket.recv()  # Wait for connection

                conversation_context = []
                memory_retention_scores = []

                for turn_num, turn in enumerate(conversation_flow):
                    turn_metrics = await self.process_conversation_turn(
                        websocket, turn, turn_num, conversation_context
                    )

                    metrics["turn_metrics"].append(turn_metrics)
                    memory_retention_scores.append(turn_metrics["memory_retention"])

                    self.logger.info(
                        f"Turn {turn_num + 1}: TTFT={turn_metrics['ttft']:.3f}s, "
                        f"Memory={turn_metrics['memory_retention']:.1f}%"
                    )

                # Calculate additional metrics
                metrics["memory_usage_trend"] = memory_retention_scores
                metrics["peak_memory_retention"] = max(memory_retention_scores)
                metrics["context_degradation_rate"] = self.calculate_degradation_rate(
                    memory_retention_scores
                )

                # Final comprehensive memory test
                final_memory_score = await self.comprehensive_final_memory_test(
                    websocket, conversation_context
                )
                metrics["final_context_accuracy"] = final_memory_score
                metrics["memory_success"] = final_memory_score >= self.memory_threshold

                # Calculate aggregate metrics
                metrics.update(
                    self.calculate_aggregate_metrics(metrics["turn_metrics"])
                )

        except Exception as e:
            self.logger.error(f"Memory test failed: {e}")
            metrics["error"] = str(e)

        self.results.append(metrics)
        return metrics

    async def process_conversation_turn(
        self, websocket, turn: Dict, turn_num: int, conversation_context: List[Dict]
    ) -> Dict[str, Any]:
        """Process a single conversation turn"""
        turn_metrics = {
            "turn_number": turn_num + 1,
            "user_message": turn["user"],
            "expected_context": turn.get("context", []),
            "ttft": None,
            "response_time": None,
            "response_quality": 0,
            "memory_retention": 0,
            "context_relevance": 0,
        }

        start_time = time.time()
        await websocket.send(turn["user"])

        response_chunks = []
        first_token = True

        while True:
            response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            data = json.loads(response)

            if data["type"] == "chunk":
                if first_token:
                    turn_metrics["ttft"] = time.time() - start_time
                    first_token = False
                response_chunks.append(data["content"])

            elif data["type"] == "complete":
                turn_metrics["response_time"] = time.time() - start_time
                full_response = "".join(response_chunks)

                # Enhanced quality analysis
                quality_metrics = self.enhanced_quality_analysis(
                    user_message=turn["user"],
                    ai_response=full_response,
                    expected_context=turn.get("context", []),
                    previous_context=conversation_context,
                    current_turn=turn_num,
                )

                turn_metrics.update(quality_metrics)

                # Update conversation context
                conversation_context.append(
                    {
                        "user": turn["user"],
                        "assistant": full_response,
                        "turn": turn_num + 1,
                    }
                )
                break

        return turn_metrics

    def enhanced_quality_analysis(
        self,
        user_message: str,
        ai_response: str,
        expected_context: List[str],
        previous_context: List[Dict],
        current_turn: int,
    ) -> Dict[str, float]:
        """Enhanced analysis with temporal context weighting"""
        quality_metrics = {
            "relevance": 1.0,
            "memory_retention": 0.0,
            "context_relevance": 0.0,
            "temporal_coherence": 1.0,
            "overall": 0.0,
        }

        # Temporal context weighting (recent context matters more)
        context_weights = self.calculate_temporal_weights(len(previous_context))

        # Check memory retention with temporal weighting
        memory_score = 0.0
        total_weight = 0.0

        for i, ctx in enumerate(previous_context):
            weight = context_weights[i] if i < len(context_weights) else 0.1
            context_text = ctx["user"] + " " + ctx["assistant"]
            keywords = self.extract_keywords(context_text)

            hits = sum(
                1 for keyword in keywords if keyword.lower() in ai_response.lower()
            )
            if keywords:
                memory_score += (hits / len(keywords)) * weight
                total_weight += weight

        if total_weight > 0:
            quality_metrics["memory_retention"] = memory_score / total_weight

        # Check expected context relevance
        expected_hits = sum(
            1
            for expected in expected_context
            if expected.lower() in ai_response.lower()
        )
        if expected_context:
            quality_metrics["context_relevance"] = expected_hits / len(expected_context)

        # Overall weighted score
        quality_metrics["overall"] = (
            quality_metrics["relevance"] * 0.3
            + quality_metrics["memory_retention"] * 0.4
            + quality_metrics["context_relevance"] * 0.2
            + quality_metrics["temporal_coherence"] * 0.1
        )

        return quality_metrics

    def calculate_temporal_weights(self, num_contexts: int) -> List[float]:
        """Calculate weights for temporal context (recent contexts weigh more)"""
        if num_contexts == 0:
            return []

        weights = [0.5 ** (num_contexts - i) for i in range(num_contexts)]
        total = sum(weights)
        return [w / total for w in weights] if total > 0 else weights

    def calculate_degradation_rate(self, memory_scores: List[float]) -> float:
        """Calculate how quickly memory retention degrades over time"""
        if len(memory_scores) < 2:
            return 0.0

        x = np.arange(len(memory_scores))
        y = np.array(memory_scores)

        # Linear regression to find slope
        slope = np.polyfit(x, y, 1)[0]
        return abs(slope)  # Return absolute value of degradation rate

    async def comprehensive_final_memory_test(
        self, websocket, conversation_context: List[Dict]
    ) -> float:
        """Test memory across different time spans in conversation"""
        if len(conversation_context) < 3:
            return 1.0

        test_scores = []

        # Test early memory (first 25%)
        early_idx = max(1, len(conversation_context) // 4)
        early_score = await self.test_specific_memory(
            websocket,
            conversation_context,
            0,
            early_idx,
            "What was the initial topic we discussed at the beginning of our conversation?",
        )
        test_scores.append(early_score)

        # Test middle memory (25-50%)
        mid_start = early_idx
        mid_end = len(conversation_context) // 2
        if mid_end > mid_start:
            mid_score = await self.test_specific_memory(
                websocket,
                conversation_context,
                mid_start,
                mid_end,
                "Can you recall what we talked about in the middle part of our conversation?",
            )
            test_scores.append(mid_score)

        # Test recent memory (last 25%)
        recent_start = len(conversation_context) * 3 // 4
        recent_score = await self.test_specific_memory(
            websocket,
            conversation_context,
            recent_start,
            len(conversation_context),
            "What were the most recent topics we discussed?",
        )
        test_scores.append(recent_score)

        return np.mean(test_scores) if test_scores else 0.0

    async def test_specific_memory(
        self,
        websocket,
        context: List[Dict],
        start_idx: int,
        end_idx: int,
        question: str,
    ) -> float:
        """Test memory for a specific conversation segment"""
        await websocket.send(question)
        response_chunks = []

        while True:
            response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            data = json.loads(response)

            if data["type"] == "chunk":
                response_chunks.append(data["content"])
            elif data["type"] == "complete":
                final_response = "".join(response_chunks)

                # Extract keywords from the target context segment
                target_context = context[start_idx:end_idx]
                all_keywords = []
                for ctx in target_context:
                    all_keywords.extend(
                        self.extract_keywords(ctx["user"] + " " + ctx["assistant"])
                    )

                # Calculate recall score
                if all_keywords:
                    unique_keywords = list(set(all_keywords))
                    hits = sum(
                        1
                        for keyword in unique_keywords
                        if keyword.lower() in final_response.lower()
                    )
                    return hits / len(unique_keywords)
                return 0.0

    def calculate_aggregate_metrics(self, turn_metrics: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate metrics from turn-level data"""
        if not turn_metrics:
            return {}

        return {
            "average_ttft": np.mean(
                [t["ttft"] for t in turn_metrics if t["ttft"] is not None]
            ),
            "average_response_time": np.mean(
                [
                    t["response_time"]
                    for t in turn_metrics
                    if t["response_time"] is not None
                ]
            ),
            "average_memory_retention": np.mean(
                [t["memory_retention"] for t in turn_metrics]
            ),
            "average_response_quality": np.mean(
                [t["response_quality"] for t in turn_metrics]
            ),
            "min_memory_retention": min([t["memory_retention"] for t in turn_metrics]),
            "max_memory_retention": max([t["memory_retention"] for t in turn_metrics]),
        }

    def analyze_scenario_results(self, scenario_results: List[Dict]) -> Dict[str, Any]:
        """Analyze results across multiple iterations of the same scenario"""
        return {
            "scenario_name": scenario_results[0]["test_id"].split("_iter")[0],
            "iterations": len(scenario_results),
            "success_rate": sum(1 for r in scenario_results if r["memory_success"])
            / len(scenario_results),
            "average_conversation_length": np.mean(
                [r["conversation_length"] for r in scenario_results]
            ),
            "average_final_accuracy": np.mean(
                [r["final_context_accuracy"] for r in scenario_results]
            ),
            "average_degradation_rate": np.mean(
                [r["context_degradation_rate"] for r in scenario_results]
            ),
            "performance_metrics": {
                "avg_ttft": np.mean(
                    [r.get("average_ttft", 0) for r in scenario_results]
                ),
                "avg_memory_retention": np.mean(
                    [r.get("average_memory_retention", 0) for r in scenario_results]
                ),
            },
        }

    def generate_memory_policy_recommendations(
        self, scenario_analyses: List[Dict]
    ) -> List[str]:
        """Generate memory policy recommendations based on benchmark results"""
        recommendations = []

        for analysis in scenario_analyses:
            if analysis["average_degradation_rate"] > 0.1:
                recommendations.append(
                    f"High context degradation ({analysis['average_degradation_rate']:.3f}) in scenario '{analysis['scenario_name']}'. "
                    "Consider implementing conversation summarization or increasing context window."
                )

            if analysis["average_final_accuracy"] < 0.6:
                recommendations.append(
                    f"Low final memory accuracy ({analysis['average_final_accuracy']:.1%}) in scenario '{analysis['scenario_name']}'. "
                    "Recommend implementing explicit memory prompts or context reinforcement."
                )

        return recommendations

    async def cleanup_session(self, session_id: str):
        """Clean up session resources"""
        try:
            # Implement session cleanup if your API supports it
            pass
        except Exception as e:
            self.logger.warning(f"Session cleanup failed for {session_id}: {e}")

    def extract_keywords(self, text: str) -> List[str]:
        """Enhanced keyword extraction"""
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "and",
            "or",
            "but",
            "this",
            "that",
        }
        words = text.lower().split()
        # Filter meaningful words and remove duplicates
        keywords = list(
            set(
                [
                    word
                    for word in words
                    if len(word) > 3 and word not in stop_words and word.isalpha()
                ]
            )
        )
        return keywords[:8]  # Return top 8 keywords


# Usage example
async def main():
    benchmark = EnhancedMemoryHistoryBenchmark()

    # Define test scenarios
    test_scenarios = [
        {
            "name": "Simple Q&A Memory",
            "conversation_flow": [
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
            "conversation_flow": [
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
            "conversation_flow": [
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

    results = await benchmark.run_comprehensive_memory_test(test_scenarios)
    print("Benchmark results:", json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
