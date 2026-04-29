from fastapi import APIRouter, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.sse import EventSourceResponse
import asyncio

from database.redis_manager import SessionKeyMap, RedisManager
from api.management.context_specs import ListenerContext
from api.management.message_specs import PublishForm

async def broadcast_generator(session_id: str, redis_manager: RedisManager):
    try:
        redis_keymap = SessionKeyMap(session_id)
        context = ListenerContext(session_id=session_id, redis_keymap=redis_keymap)
        context.stream_state[redis_keymap.stream_transcription] = "0-0"
        context.stream_state[redis_keymap.stream_translation] = "0-0"
        
        # session이 종료되었는지 확인
        # 종료되지 않았다면 제네레이터를 반환
        # 종료되었다면 db에서 데이터를 리턴
        # db에 데이터가 저장되어있는지 여부
        try: 
            message = PublishForm(type="init")
            bundle = await redis_manager.read_stream(context.stream_state, block=1000)

            for stream in bundle:
                key, items = stream
                last_id = items[-1][0]
                context.stream_state[key] = last_id

                if key == redis_keymap.stream_transcription:
                    message.payload["transcription"] = [item[1] for item in items if item[1]["complete"] == "true"]
                if key == redis_keymap.stream_translation:
                    message.payload["translation"] = [item[1] for item in items]
            
            if bundle: 
                yield f"data: {message.model_dump_json()}\n\n"

        except Exception as e:
            print("listener init error:", e)

        while True:
            try:
                bundle = await redis_manager.read_stream(context.stream_state, block=3000)
                if not bundle:
                    continue

                message = PublishForm(type="next")
                for stream in bundle:
                    key, items = stream
                    last_id = items[-1][0]
                    context.stream_state[key] = last_id

                    if key == redis_keymap.stream_transcription: message.payload["transcription"] = items[-1][1]
                    if key == redis_keymap.stream_translation: message.payload["translation"] = items[-1][1]

                yield f"data: {message.model_dump_json()}\n\n"

            except Exception as e:
                print("listener while error:", e)
            else:
                pass
            finally:
                pass

    except Exception as e:
        print("listener end error:", e)
    
    finally:
        message = PublishForm(type="done")
        yield f"data: {message.model_dump_json()}\n\n"

