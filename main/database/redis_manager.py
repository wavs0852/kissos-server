from dataclasses import dataclass, field
import redis.asyncio as aioredis
from typing import Any
import functools


""" 
KeyMap
- ManageKeyMap: Keys for global management 
- SessionKeyMap: Keys for session-specific operations
"""

@dataclass
class ManageKeyMap:
    prefix: str = "manage"
    set_active_session: str = f"{prefix}:set:active-session"
    set_inactive_session: str = f"{prefix}:set:inactive-session"
    # active_client: str
    # client_session

@dataclass
class SessionKeyMap:
    session_id: str
    prefix: str = field(init=False)
    hash_info: str = field(init=False)
    pubsub_manage: str = field(init=False)
    stream_transcription: str = field(init=False)
    stream_translation: str = field(init=False)

    def __post_init__(self):
        self.prefix = f"session:{self.session_id}"
        self.hash_info = f"{self.prefix}:hash:info"
        self.pubsub_manage = f"{self.prefix}:pubsub:system"
        self.stream_transcription = f"{self.prefix}:stream:trscrpt"
        self.stream_translation = f"{self.prefix}:stream:trslt"
        


""" 
RedisManager 
A wrapper around aioredis to manage Redis connections and operations

- connect: Establish connection to Redis server
- disconnect: Close connection to Redis server
- cleanup: Flush the Redis database (use with caution)
- update_stream: Add a message to a stream
- update_pubsub: Publish a message to a channel
- update_hash: Set fields in a hash
- read_stream: Read from streams with optional blocking and count
- read_pubsub: Subscribe to channels and return a PubSub object for listening
"""

def error_handler(f):
    @functools.wraps(f)
    async def wrapper(self, *args, **kwargs):
        try:
            return await f(self, *args, **kwargs)
        except Exception as e:
            print(f"[Error] Redis {e}")
            return None 
    return wrapper

class RedisManager:
    def __init__(self):
        # self.client: aioredis.Redis | None = None
        self.client = aioredis.Redis(
            host='localhost',
            port=6380,
            db=0,
            decode_responses=True
        )
        self.cached_logs = {}

    # async def connect(self):
    #     print("RedisManager connecting...")
    #     if not self.client:
    #         self.client = aioredis.Redis(
    #             host='localhost',
    #             port=6380,
    #             db=0,
    #             decode_responses=True
    #         )

    async def disconnect(self):
        if self.client:
            await self.client.close()
            self.client = None

    async def cleanup(self):
        await self.client.flushdb()
    
    @error_handler
    async def update_stream(self, key, message, *args, **kwargs):
        await self.client.xadd(key, message, *args, **kwargs)

    @error_handler
    async def update_hash(self, key, message, *args, **kwargs):
        await self.client.hset(key, mapping=message, *args, **kwargs)

    @error_handler
    async def read_stream(self, streams, count=None, block=None):
        result = await self.client.xread(streams, count=count, block=block)
        return result
    

    @error_handler
    async def update_pubsub(self, key, message, *args, **kwargs):
        await self.client.publish(key, message, *args, **kwargs)

    async def read_pubsub(self, channels):
        pubsub = self.client.pubsub()
        await pubsub.subscribe(*channels)
        return pubsub



