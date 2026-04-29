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
    for i in range(5):
        await asyncio.sleep(1)
        a = time.time()
        await client.xadd("test_1", {"message": str(i)})
        # await client.xadd("test_2", {"message": str(i*10)})
        print(f"Message {i} added to stream. Time taken: {time.time() - a} seconds")
    print("add End")


async def read():
    test_1_idx = "0-0"
    test_2_idx = "0-0"
    await asyncio.sleep(3)
    a = time.time()
    messages = await client.xread({
        "test_1": test_1_idx,
        # "test_2": test_2_idx
        })
    print("Messages read from stream:", messages)
    print(f"Time taken to read messages: {time.time() - a} seconds")
    # test_1_idx = messages[0][1][-1][0]
    # test_2_idx = messages[1][1][-1][0]
    await asyncio.sleep(2)
    messages = await client.xread({
        "test_1": test_1_idx,
        # "test_2": test_2_idx
        })
    print("Messages read from stream:", messages)
    print(f"Time taken to read messages: {time.time() - a} seconds")

# 반환 리스트 len으로 
# num_of_streams = len(messages)
# if num_of_streams == 1:
#     if messages[0][0] == "test_1":
#         transcription = messages[0][1]
#         translation = None

# elif num_of_streams == 2:
#     transcription = mesasages[0][1]
#     translation = messages[1][1]

async def doit():
    await client.flushdb()  # Clear the database before starting
    print("gather 시작")
    results = await asyncio.gather(
        add(),
        read()
    )

if __name__ == "__main__":
    
    asyncio.run(doit())