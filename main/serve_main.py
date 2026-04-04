import re
import json
import asyncio
from dataclasses import dataclass, field

import redis.asyncio as redis
import numpy as np

import ray
from ray import serve

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse

from constants import LANG_IDS
from client_manager import ClientManager
from process_manager import TranscriptionState
from transcription_deployment import DeployWhisper


ray.init(_temp_dir="/data/tmp", ignore_reinit_error=True)
app = FastAPI()


@serve.deployment
@serve.ingress(app)
class IngressServer:
    def __init__(self, transcription_handle):
        self.transcription_handle = transcription_handle 
        self.translation_handle = None

        self.r = redis.Redis(host='localhost', port=6380, db=0, decode_responses=True) 
        self.r.flushdb()    

    @app.websocket("/api/streaming")
    async def echo(self, websocket: WebSocket):
        # 사용자 인증 로직 + 타임아웃 설정
        try:
            await websocket.accept()
            print("Websocket connection accepted")
            dummy_session_id = "369369"
            manager = ClientManager(session_id=dummy_session_id)
        except Exception as e:
            print(f"Websocket initial connection failed: {e}")
            return # 연결 실패 시 즉시 종료

        # 엔진 내부의 에러 핸들링
        # 버퍼 핸들링

        async def socket_loop():
            try:
                try:
                    user_vars = await asyncio.wait_for(websocket.receive_json(), timeout=1.0)

                    transcription_languages = user_vars["payload"]["languages"]
                    translation_languages = None 
                    manager.transcription_languages = [LANG_IDS[lang] for lang in transcription_languages]
                    manager.translation_languages = None

                except (asyncio.TimeoutError, KeyError, json.JSONDecodeError) as e:
                    print(f"Initial setup failed: {e}")
                    manager.connection_disconnected = True
                    return # 초기 설정 실패 시 즉시 종료

                while not manager.connection_disconnected:
                    try:
                        data = await websocket.receive_bytes()
                        await manager.audio_queue.put(data)
                        manager.left_audio_for_trscrpt += 1

                    except WebSocketDisconnect as e:
                        # 정상/비정상 종료 구분
                        if e.code in [1000, 1001]:
                            print(f"Client disconnected normally: {e.code}")
                        else:
                            print(f"Client connection lost unexpectedly: {e.code}")
                        break # 루프 탈출
                    except Exception as e:
                        print(f"Unexpected error in socket_loop: {e}")
                        break
            finally:
                manager.connection_disconnected = True
                await manager.audio_queue.put(None)
                print("Socket loop ended")
                    
        async def transcription_loop(): 
            # NOTE: queue의 오버플로우와 이를 처리하기 위한 순서 있는 청크 로직이 부재한 naive한 구현임
            try:
                state = TranscriptionState()

                while True:
                    if manager.connection_disconnected and manager.left_audio_for_trscrpt == 0:
                        manager.audio_queue.task_done()
                        print("No more audio to process and client disconnected. Ending transcription loop.")
                        break

                    data = await manager.audio_queue.get()
                    if data is None and manager.left_audio_for_trscrpt == 0: # 소켓 루프에서 종료 신호로 None을 보냄
                        manager.audio_queue.task_done()
                        break
                    islong = data[0] == 1
                    isedge = data[1] == 1
                    stream = np.frombuffer(data, dtype=np.float32, offset=2)
                    
                    # [Transcription]
                    if len(stream) > 1:
                        self.handle_transcription_prompt(state, islong, isedge)

                        transcription_result = await self.transcription_handle.generate.remote(
                            encoder_input=stream,
                            decoder_input=state.prompt, 
                            allowed_languages=manager.transcription_languages, 
                            # redis_keys=manager.redis_keys, 
                            )
                        
                        print(transcription_result)
                    
                        text_result = re.sub(r"<\|.*?\|>", "", transcription_result.outputs[0].text)
                        # update state
                        state.buffer = f"{state.buffer}{text_result}" if islong else text_result
                        state.num_of_words_history.append(state.buffer.count(' ') + 1) # 업데이트할때마다 안에서 쌓이기

                    if isedge: 
                        print(f"say: {state.buffer} (finished)")
                        # await self.r.xadd(history_key, {"text": state.buffer}, maxlen=2000)
                        state = TranscriptionState()
                        
                    else:
                        print(f"say: {state.buffer} (unfinished)")
                        # await self.r.xadd(live_key, {"text": state.buffer}, maxlen=1)

                    manager.left_audio_for_trscrpt -= 1
                    manager.audio_queue.task_done()
            except Exception as e:
                print(f"Unexpected error in socket_loop: {e}")
                
            finally:
                print("Transcription loop ended.")

        tasks = [
            asyncio.create_task(socket_loop()),
            asyncio.create_task(transcription_loop())
        ]


        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in pending:
            if not task.done():
                try:

                    await asyncio.wait_for(task, timeout=10.0)
                except asyncio.TimeoutError:
                    task.cancel() # 끝까지 안 나오면 강제 종료





    @staticmethod
    def handle_transcription_prompt(state, islong, isedge):
        # TODO (Space 기반) 단어 수의 미스매치 수정 / (Char 기반) 구현하기
        def find_nth_space(text, n, offset=-1):
            for _ in range(n):
                offset = text.find(' ', offset + 1)
                if offset == -1: break
            return offset
        
        update_count = len(state.num_of_words_history)
        if islong:
            if isedge:
                # TODO Char 기반 대응 (중국어/일본어 등)
                prev_prompt_endpoint = state.num_of_words_history[-3]
                prev_prompt_endpoint = find_nth_space(state.buffer, prev_prompt_endpoint)
            else:
                last_prev_update_idx = update_count-1 if update_count % 2 else update_count-2
                prev_prompt_endpoint = state.num_of_words_history[last_prev_update_idx-1]
                prev_prompt_endpoint = find_nth_space(state.buffer, prev_prompt_endpoint)

            prev_prompt_startpoint = 0 if prev_prompt_endpoint < 10 else prev_prompt_endpoint - 10
            prev_prompt = state.buffer[prev_prompt_startpoint:prev_prompt_endpoint]

            state.prompt = f"<|startofprev|>{prev_prompt}<|startoftranscript|>" #{state.lang_token}<|transcribe|><|notimestamps|>"
            state.buffer = state.buffer[:prev_prompt_endpoint]


transcription_handle = DeployWhisper.bind()
runner = IngressServer.bind(transcription_handle)