import asyncio
from functools import partial

import ray
from ray import serve

from fastapi import Depends, FastAPI, WebSocket, Request
from fastapi.sse import EventSourceResponse
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from deployment.transcriptor import DeployWhisper
from deployment.translator import DeployQwen
from database.redis_manager import RedisManager
from auth.oauth import router as auth_router
from api import speaker, listener
from api.management.manager import ConnectionManger

ray.init(_temp_dir="/data/tmp", ignore_reinit_error=True)



# app = FastAPI()
# app.include_router(auth_router) # 지울 예정

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis_manager = RedisManager()
    # await redis_manager.connect()
    await app.state.redis_manager.cleanup()
    yield
    await app.state.redis_manager.disconnect()



app = FastAPI(lifespan=lifespan)
app.include_router(auth_router)

@serve.deployment
@serve.ingress(app)
class IngressServer:
    def __init__(self, transcriptor_handle, translator_handle):
        self.transcriptor_handle = transcriptor_handle
        self.translator_handle = translator_handle
        self.redis_manager = RedisManager()
        self.manager = ConnectionManger()

        # app.router.lifespan_context = self.lifespan
        
    # @asynccontextmanager
    # async def lifespan(self, app: FastAPI):
    #     self.redis_manager.connect()
    #     self.redis_manager.cleanup()
    #     yield
    #     self.redis_manager.disconnect()

    # route prefix ?ъ슜
    @app.websocket("/api/streaming")
    async def websocket_endpoint(self, websocket: WebSocket):
        print("GOT A CONNECTION")
        print(app.state.redis_manager)
        await speaker.stream(
            websocket,
            app.state.redis_manager,
            self.transcriptor_handle, 
            self.translator_handle,
            )

    @app.get("/api/broadcast/{session_id}")
    async def websocket_endpoint(self, request: Request, session_id: str):
        return EventSourceResponse(
            listener.broadcast_generator(
                session_id,
                app.state.redis_manager
            ))
        

app_transcriptor = DeployWhisper.bind()
app_translator = DeployQwen.bind()
app_ingress = IngressServer.bind(app_transcriptor, app_translator)


#    @app.get("/api/broadcast/{session_id}")
#     async def websocket_endpoint(self, request: Request, session_id: str):
#         # async for result in broadcast.worker(
#         #     request,
#         #     session_id,
#         #     self.redis_client
#         #     ):
#         #     yield result
