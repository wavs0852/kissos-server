import redis.asyncio as aioredis
import asyncio
import time
client = aioredis.Redis(
    host='localhost',
    port=6380,
    db=0,
    decode_responses=True
    )


async def add():
    for i in range(5000):
        # await asyncio.sleep(0.2)
        a = time.time()
        await client.xadd("test1", {"message": str(i), "type": "h"})
        await client.xadd("test2", {"message": str(i*10)})
        # print(f"Message {i} added to stream. Time taken: {time.time() - a} seconds")
    print("add End")


async def read():
    await asyncio.sleep(3)
    a = time.time()
    messages = await client.xread({"test1": "0-0", "test2": "0-0"})
    print("Messages read from stream:", messages)
    print(f"Time taken to read messages: {time.time() - a} seconds")

    for msg in messages:
        # print(msg[0], ":", msg[1])
        a = time.time()
        u = [m for m in msg[1] if m[1].get("type") == "h"]
        print(f"Time taken to filter messages: {time.time() - a} seconds")
    # await asyncio.sleep(2)
    # messages = await client.xread({"test_1": test_1_idx
    #     })
    # print("Messages read from stream:", messages)
    # print(f"Time taken to read messages: {time.time() - a} seconds")



async def doit():
    await client.flushdb()  # Clear the database before starting
    print("gather 시작")
    results = await asyncio.gather(
        add(),
        read()
    )

if __name__ == "__main__":
    
    asyncio.run(doit())