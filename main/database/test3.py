from redis_manager import RedisManager, SessionKeyMap, MessageForm
import asyncio

redismanager = RedisManager()
redis_keymap = SessionKeyMap(session_id="123456")

async def add():
    for i in range(5):
        transcription = MessageForm(
            type="transcription",
            number=i,
            complete=1,
            language="en",
            result="Hello, World!"
        )
        translation = MessageForm(
            type="translation",
            number=i,
            complete=1,
            language="ko",
            result="안녕하세요, 세계!"
            )


        await redismanager.update_stream(redis_keymap.stream_transcription, transcription.model_dump())
        await redismanager.update_stream(redis_keymap.stream_translation, translation.model_dump())

# 우선 TTL로 구현 | 종료된 세션은 TTL 없이 바로 만료 > 데이터베이스 이관
# 세션이 종료되었는지 아닌지 알기 위해서는 hash_info에 상태를 업데이트해야함
# 세션 만료 시간을 프론트엔드에 표시해야함 
from typing import Any, Union

async def read():
    await asyncio.sleep(1)
    messages  = await redismanager.read_stream({
        redis_keymap.stream_transcription: "0-0",
        redis_keymap.stream_translation: "0-0"
    })
    # print(messages)

    structure = {}
    for msg in messages:
        
        print(msg[0], msg[1][-1][0])
        # print(msg[1])
        print([each[1] for each in msg[1]])




async def doit():
    await redismanager.cleanup()
    await asyncio.gather(
        add(),
        read()
    )

if __name__ == "__main__":
    asyncio.run(doit())