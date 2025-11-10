#!/usr/bin/env python3
import asyncio
import logging
import os
import sys
import time
from typing import Any, Dict
from uuid import uuid4
import redis.asyncio as redis
from chat_agent import InferenceService
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_architect.datatype_abstraction import Features, TextFeatures
from agent_architect.session_abstraction import AgentSessions, SessionStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sids = []
# Redis configuration (must match your RAG service)
INPUT_CHANNELS = ["RAG:high", "RAG:low"]
AGENT_NAME = "chat"

OUTPUT_CHANNEL = f"{AGENT_NAME.lower()}:output"
OUTPUT_CHANNELS = [OUTPUT_CHANNEL]
ACTIVE_SESSIONS_KEY = "chat:active_sessions"

SERVICE_NAMES = ["RAG"]
CHANNEL_STEPS = {
    "RAG": ["high", "low"],
}
INPUT_CHANNEL = f"{SERVICE_NAMES[0]}:{CHANNEL_STEPS[SERVICE_NAMES[0]][0]}"

chat_agent = InferenceService(
    agent_type="chat",
    service_names=["RAG"],
    channels_steps={"RAG": ["high", "low"]},
    input_channel="RAG:low",
    output_channel="chat:output",
    timeout=60.0,
)


async def publish_rag_requests(redis_client, num_requests: int = 2):
    global sids
    sids = [f"{'Fadi'}:{'002'}:{i}:{str(uuid4().hex)}" for i in range(num_requests)]

    # Mark sessions as active
    for sid in sids:
        status = AgentSessions(
            sid=sid,
            agent_type=AGENT_NAME,
            agent_id="AGENT_ID",
            service_names=SERVICE_NAMES,
            channels_steps=CHANNEL_STEPS,
            owner_id="+12345952496",
            status=SessionStatus.ACTIVE,
            first_channel=INPUT_CHANNEL,
            last_channel=OUTPUT_CHANNEL,
            timeout=3000,
        )

        await redis_client.hset(ACTIVE_SESSIONS_KEY, sid, status.to_json())

    txt_objects = [
        TextFeatures(
            sid=sids[0],
            priority="high",
            created_at=time.time(),
            text="Hello. My name is Borhan. Who are you?",
            agent_type='chat',
            is_final=False,
        ),
        TextFeatures(
            sid=sids[1],
            priority="high",
            created_at=time.time(),
            text="Hi my name is Jack, what's your name?",
            agent_type='chat',
            is_final=False,
        ),
        # TextFeatures(
        #     sid="test_sid_4",
        #     priority="high",
        #     created_at=time.time(),
        #     text="I love music.",
        # ),
        # TextFeatures(
        #     sid="test_sid_2",
        #     priority="high",
        #     created_at=time.time(),
        #     text="He ate a small apple.",
        # ),
        # TextFeatures(
        #     sid="test_sid_0",
        #     priority="high",
        #     created_at=time.time(),
        #     text="They walked to this door.",
        # ),
        # TextFeatures(
        #     sid="test_sid_8",
        #     priority="high",
        #     created_at=time.time(),
        #     text="The book is very old.",
        # ),
        # TextFeatures(
        #     sid="test_sid_6",
        #     priority="high",
        #     created_at=time.time(),
        #     text="Hello, I am Jack.",
        # ),
        # TextFeatures(
        #     sid="test_sid_4",
        #     agent_type="call",
        #     priority="high",
        #     created_at=time.time(),
        #     text="I love music.",
        # ),
        # TextFeatures(
        #     sid="test_sid_2",
        #     priority="high",
        #     created_at=time.time(),
        #     text="He ate a small apple.",
        # ),
        # TextFeatures(
        #     sid="test_sid_0",
        #     priority="high",
        #     created_at=time.time(),
        #     text="They walked to this door.",
        # ),
    ]
    
    tasks = []
    for i in txt_objects:
        channel = f"RAG:{i.priority}"

        logger.info(
            f"Publishing RAG request (sid={i.sid}, priority={i.priority}) to {channel}"
        )
        tasks.append(redis_client.lpush(channel, i.to_json()))

    await asyncio.gather(*tasks)
    logger.info(f"Published {len(txt_objects)} RAG requests.")


# async def listen_for_rag_results(redis_client, expected_count: int, timeout: int = 60):

#     # raw = await redis_client.hget(ACTIVE_SESSIONS_KEY, sid)
#     # if raw is None:
#     #     return None
#     # status_obj = AgentSessions.from_json(raw)
#     result = await redis_client.brpop(OUTPUT_CHANNEL, timeout=0)
    
#     # pubsub = redis_client.pubsub()
#     # await pubsub.subscribe(*OUTPUT_CHANNELS)

#     results = []
#     start_time = asyncio.get_event_loop().time()

#     logger.info("Listening for RAG results...")
#     async for message in pubsub.listen():
#         # if message["type"] != "message":
#         #     continue
#         print(f"Received message: {message}")
#         channel = message["channel"].decode()
#         logger.info(f"✅ Received RAG result on {channel}")
#         results.append(message["data"])

#         if len(results) >= expected_count:
#             break

#         if asyncio.get_event_loop().time() - start_time > timeout:
#             logger.warning("Timeout reached while waiting for RAG results.")
#             break

#     await pubsub.unsubscribe(*OUTPUT_CHANNELS)
#     return results
async def listen_for_rag_results(redis_client, expected_count: int, timeout: int = 60):
    results = []
    start_time = asyncio.get_event_loop().time()

    logger.info("Listening for RAG results...")
    
    while len(results) < expected_count:
        current_time = asyncio.get_event_loop().time()
        remaining_time = int(timeout - (current_time - start_time))
        
        if remaining_time <= 0:
            logger.warning("Timeout reached while waiting for RAG results.")
            break
            
        result = await redis_client.brpop(OUTPUT_CHANNEL, timeout=remaining_time)
        print(f"Received result: {result}")
        if result is None:
            logger.warning("Timeout reached while waiting for RAG results.")
            break
            
        _, data = result
        logger.info(f"✅ Received RAG result on {OUTPUT_CHANNEL}")
        results.append(data)

    return results

async def main():
    NUM_REQUESTS = 5
    TIMEOUT = 60

    redis_client = await redis.from_url(
        "redis://localhost:6379", decode_responses=False
    )

    # Start listener before publishing
    listener_task = asyncio.create_task(
        listen_for_rag_results(redis_client, NUM_REQUESTS, TIMEOUT)
    )

    # Publish simulated RAG outputs
    await publish_rag_requests(redis_client, num_requests=NUM_REQUESTS)

    # Wait for results
    try:
        results = await asyncio.wait_for(listener_task, timeout=TIMEOUT)
        logger.info(f"✅ Received {len(results)} / {NUM_REQUESTS} RAG results.")
    except asyncio.TimeoutError:
        logger.error("❌ Timeout: Not all RAG results received.")
        results = listener_task.result() if listener_task.done() else []

    # Cleanup: mark sessions as stopped
    # sids = [f"test_sid_{i}" for i in range(NUM_REQUESTS)]
    # for sid in sids:
    #     print(f"Marking session {sid} as stopped.")
    #     stop_status = SessionStatus(
    #         sid=sid, status="stop", created_at=None, timeout=0.0
    #     )
    #     await redis_client.hset(ACTIVE_SESSIONS_KEY, sid, stop_status.to_json())

    await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
