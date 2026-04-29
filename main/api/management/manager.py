import secrets
import string
from database.redis_manager import RedisManager, ManageKeyMap


class ConnectionManger:
    def __init__(self):
        cache = {}
        backgroud_task = set()

    async def get_session_id(redis_manager: RedisManager):
        session_id = await redis_manager.spop("m:sessions:inactive")
        if not session_id:
            while True:
                session_id = ''.join(secrets.choice(string.digits) for _ in range(6))
                if not await redis_manager.sismember("m:sessions:active", session_id):
                    break
            
        await redis_manager.sadd(f"m:sessions:active", session_id)

        return session_id
    
    async def end_session(redis_manager: RedisManager, session_id: str):
        await redis_manager.srem(f"m:sessions:active", session_id)
        await redis_manager.sadd(f"m:sessions:inactive", session_id)

    # 세션 종료에 따라 listner에게 종료 신호 보내기 + 데이터 이관, listener는 세션 종료 신호를 받으면 남아있는 데이터를 db에서 가져가도록 하기
    # TTL 설정해서 일정 시간 후에 자동으로 데이터 삭제되도록 하기
    async def notify_session_end(redis_manager: RedisManager, session_id: str):
        keymap = ManageKeyMap()
        await redis_manager.publish(keymap.set_inactive_session, session_id)