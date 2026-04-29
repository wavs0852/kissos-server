import asyncio
import json
import numpy as np
import re
import secrets
import string
import traceback
from fastapi import APIRouter, Request, Depends, WebSocket, WebSocketDisconnect
from typing import Any, Dict, List
from ray.serve.handle import DeploymentHandle

from constant.whisper import WHISPER_LANG_IDS_REVERSED, WHISPER_LANG_IDS, WHISPER_LANG_ID_TAGS
from auth.oauth import verify_kissos_token
from database.redis_manager import SessionKeyMap, RedisManager
from api.management.context_specs import SpeakerContext, TranscriptionState
from api.management.message_specs import TranscriptionMessage, TranslationMessage


async def get_session_id(redis):
    session_id = await redis.spop("m:sessions:inactive")

    if not session_id:
        while True:
            session_id = ''.join(secrets.choice(string.digits) for _ in range(6))
            if not await redis.sismember("m:sessions:active", session_id):
                break
        
    await redis.sadd(f"m:sessions:active", session_id)

    return session_id

async def stream(
    websocket: WebSocket,
    redismanager: RedisManager,
    transcription_handle: DeploymentHandle,
    translator_handle: DeploymentHandle
    ):

    try:
        # WebSocket 연결 직후 인증
        print("inside stream")
        await websocket.accept()
        print("WebSocket connection accepted, awaiting authentication...")
        user_vars = await asyncio.wait_for(websocket.receive_json(), timeout=3.0)
        # # accessToken 꺼내기
        payload = user_vars.get("payload", {})
        # access_token = payload.get("accessToken")
        # if not access_token:
        #     await websocket.send_json({"type": "auth_error", "message": "Missing access token"})
        #     await websocket.close(code=1008)
        #     return
        # # JWT 검증
        # auth_claims = verify_kissos_token(access_token)
    except Exception as e:
        print(f"Websocket initial/auth connection failed: {e}")
        try:
            await websocket.send_json({"type": "auth_error", "message": "Authentication failed"})
            await websocket.close(code=1008)
        except Exception:
            pass
        return
    
    # 인증 성공 후 session 생성
    try:
        # session_id = await get_session_id(redis)
        session_id = "369369"
        redis_keymap = SessionKeyMap(session_id)
        context = SpeakerContext(session_id, redis_keymap)

        # Redis에 인증 정보 저장
        # await redismanager.update_hash(
        #     redis_keymap.hash_info, {
        #         "pin": session_id,
        #         "auth_sub": auth_claims.get("sub", ""),
        #         "auth_provider": auth_claims.get("provider", "")
        #         }
        # )
        # 클라이언트에 session 정보 전달
        await websocket.send_json({
            "type": "session",
            "payload": {
                "pin": session_id,
                "sessionId": session_id,
                "broadcastUrl": f"/api/broadcast/{session_id}",
            }
        })
        
        languages_group = payload["languages"]
        context.transcription_languages = [WHISPER_LANG_IDS[lang] for lang in languages_group]
        context.translation_languages = languages_group

        async def socket_loop():
            try:
                while not context.connection_disconnected:
                    try:
                        message = await websocket.receive()
                        print("ws type:", message.get("type"))
                        print("ws has text:", message.get("text") is not None)
                        print("ws bytes len:", None if message.get("bytes") is None else len(message["bytes"]))

                        if message["type"] == "websocket.disconnect":
                            print(f"Client disconnected: {message.get('code')}")
                            break

                        if message["type"] != "websocket.receive":
                            continue

                        data = message.get("bytes")
                        if data is None:
                            print("Received non-bytes websocket frame")
                            continue

                        await context.audio_queue.put(data)

                    except WebSocketDisconnect as e:
                        print(f"WebSocketDisconnect: code={e.code}")
                        break

                    except Exception as e:
                        print(f"Unexpected error in socket_loop: type={type(e).__name__}, detail={e!r}")
                        traceback.print_exc()
                        break
            finally:
                context.connection_disconnected = True
                await context.audio_queue.put(None)

                    
        async def transcription_loop(): 
            try:
                state = TranscriptionState()
                result = None

                while True:
                    try:
                        data = await context.audio_queue.get()
                        if data is None: # manager.audio_queue._unfinished_tasks == 1
                            context.audio_queue.task_done()
                            break
                        
                        islong = data[0] == 1
                        isend = data[1] == 1
                        stream = np.frombuffer(data, dtype=np.float32, offset=2) 
                        
                        if len(stream) > 1:
                            handle_transcription_prompt(state, islong, isend)

                            transcription_result = await transcription_handle.generate.remote(
                                encoder_input=stream,
                                decoder_input=state.prompt, 
                                allowed_languages=context.transcription_languages, 
                                )

                            text_result = re.sub(r"<\|.*?\|>", "", transcription_result.outputs[0].text)
                            print("[CLEAN TRANSCRIPTION]", text_result, flush=True)
                            
                            # update state
                            state.buffer = f"{state.buffer}{text_result}" if islong else text_result
                            state.num_of_words_history.append(state.buffer.count(' ') + 1) # ?낅뜲?댄듃?좊븣留덈떎 ?덉뿉???볦씠湲?
                            first_token = transcription_result.outputs[0].token_ids[0]
                            if first_token in WHISPER_LANG_ID_TAGS:
                                state.lang_token = first_token
                                state.lang = WHISPER_LANG_IDS_REVERSED[first_token]

                        # cur_lang = state.lang
                        # cur_text = state.buffer
                            result = {
                                "language": state.lang, 
                                "text": state.buffer
                                }
                        
                        if isend: state = TranscriptionState()
                        context.audio_queue.task_done()

                    except Exception as e:
                        print(f"Unexpected error in transcription_loop - while_loop: {e}")
                        continue

                    else:
                        try:
                            message = TranscriptionMessage(
                                type="transcription", 
                                status="success",
                                number=context.transcription_message_number, 
                                complete="true" if isend else "false",
                                language=result["language"],
                                result=result["text"],
                                )
                            
                            await redismanager.update_stream(redis_keymap.stream_transcription, message.model_dump())
                            
                            if isend: 
                                await context.trans_queue.put(result)
                                context.transcription_message_number += 1


                                print(f"say: {result} (finished)")
                            else:
                                print(f"say: {result} (unfinished)")
                        except Exception as e:
                            print(f"[ERROR] transcription_loop:else \n{e}")
                            pass

                    finally:
                        pass
                    
            except Exception as e:
                print(f"Unexpected error in transcription_loop: {e}")
                
            finally:
                print("Transcription loop ended.")
                await context.trans_queue.put(None)


        async def translation_loop():
            try:
                translation_langs = context.translation_languages or []

                if not translation_langs:
                    print(f"[CLOSE] translation_loop \nThere's no translation_langs")
                    return

                while True:
                    try:
                        data = await context.trans_queue.get()
                        if data is None and context.trans_queue._unfinished_tasks == 1: 
                            context.trans_queue.task_done()
                            break

                        transcription_lang = data["language"]
                        transcription_text = data["text"]

                        translation_langs_filtered = [lang for lang in translation_langs if lang != transcription_lang]
                        if translation_langs_filtered:
                            result = await translator_handle.generate.remote(
                                                                cur_lid=transcription_lang,
                                                                trans_langs=translation_langs_filtered,
                                                                text=transcription_text,
                                                                )
                            print("[TRANSLATION RESULT]", result, flush=True)
                        context.trans_queue.task_done()

                    except Exception as e:
                        print(f"[ERROR] translation_loop:while \n{e}")
                        continue

                    else:
                        try:
                            message = TranslationMessage(
                                type="translation", 
                                status="success",
                                number=context.translation_message_number,
                                language=", ".join(translation_langs_filtered),
                                **result
                                )
                            
                            await redismanager.update_stream(redis_keymap.stream_translation, message.model_dump())
                            context.translation_message_number += 1
                            print("translation:", result)
                        except Exception as e:
                            print(f"[ERROR] translation_loop:else \n{e}")
                            pass

                    finally:
                        pass
                        
                        

            except Exception as e:
                print(f"Unexpected error in translation_loop: {e}")
                
            finally:
                print("Translation loop ended.")


        tasks = [
            asyncio.create_task(socket_loop()),
            asyncio.create_task(transcription_loop()),
            asyncio.create_task(translation_loop())
        ]

        done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    except asyncio.TimeoutError as e:
        return
    except (KeyError, json.JSONDecodeError) as e:
        return 
    except Exception as e:
        print(f"Unexpected error in main stream function: {e}")
        traceback.print_exc()

    finally:
        # TTL / Pub/Sub 醫낅즺湲곗?
        # promice를 걸기 / pubsub으로 신호 보내기?
        return


def handle_transcription_prompt(state, islong, isend):
    # TODO (Space 湲곕컲) ?⑥뼱 ?섏쓽 誘몄뒪留ㅼ튂 ?섏젙 / (Char 湲곕컲) 援ы쁽?섍린
    def find_nth_space(text, n, offset=-1):
        for _ in range(n):
            offset = text.find(' ', offset + 1)
            if offset == -1: break
        return offset
    
    update_count = len(state.num_of_words_history)
    if islong:
        if isend:
            # TODO Char 湲곕컲 ???(以묎뎅???쇰낯????
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


    # for task in pending:
    #     if not task.done():
    #         try:
    #             await asyncio.wait_for(task, timeout=10.0)
    #         except asyncio.TimeoutError:
    #             task.cancel() # ?앷퉴吏 ???섏삤硫?媛뺤젣 醫낅즺




