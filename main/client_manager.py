from dataclasses import dataclass, field
import asyncio
import json


@dataclass
class RedisKeys:
    sid: str
    meta: str = field(init=False)
    transcription_q: str = field(init=False)
    transcription_s: str = field(init=False)
    transcription_ps: str = field(init=False)
    translation_q: str = field(init=False)
    translation_s: str = field(init=False)
    translation_ps: str = field(init=False)
    
    def __post_init__(self):
        self.meta = f"s:{self.sid}:meta"
        self.transcription_q = f"s:{self.sid}:q:trscrpt"
        self.transcription_s = f"s:{self.sid}:s:trscrpt"
        self.transcription_ps = f"s:{self.sid}:ps:trscrpt"
        self.translation_q = f"s:{self.sid}:q:trslt"
        self.translation_s = f"s:{self.sid}:s:trslt"
        self.translation_ps = f"s:{self.sid}:ps:trslt"

@dataclass
class ClientManager:
    session_id: str
    redis_keys: RedisKeys = field(init=False)
    audio_queue: asyncio.Queue = field(init=False)
    connection_disconnected: bool = False
    connection_error: bool = False # 연결에서의 에러

    transcription_languages: list = field(default_factory=list)
    translation_languages: list = field(default_factory=list)

    left_audio_for_trscrpt: int = 0 # 미처리된 오디오 # queue_unfinished_tasks로 대체 가능하지 않을까
    left_trscrpt_for_trslt: int = 0 # 미처리된 트랜스크립션
    # 처리 순서가 역전되있을 때를 대비해 정렬 리스트를 만들어두자
    # 만일 정렬 리스트에 값이 있다면 연속해서 처리하기

    def __post_init__(self):
        self.redis_keys = RedisKeys(sid=self.session_id)
        self.audio_queue = asyncio.Queue()