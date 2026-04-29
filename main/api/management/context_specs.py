import asyncio
from dataclasses import dataclass, field
from database.redis_manager import SessionKeyMap
from typing import Any, Dict, List

@dataclass
class SpeakerContext:
    session_id: str
    redis_keys: SessionKeyMap
    audio_queue: asyncio.Queue = field(init=False)
    trans_queue: asyncio.Queue = field(init=False)
    transcription_languages: List[str] = field(default_factory=list)
    translation_languages: List[str] = field(default_factory=list)

    transcription_message_number: int = 0
    translation_message_number: int = 0
    connection_disconnected: bool = False
    connection_error: bool = False 

    def __post_init__(self):
        self.audio_queue = asyncio.Queue()
        self.trans_queue = asyncio.Queue()
    # 처리 순서가 역전되있을 때를 대비해 정렬 리스트를 만들어두자
    # 만일 정렬 리스트에 값이 있다면 연속해서 처리하기

@dataclass
class TranscriptionState:
    prompt: str = "<|startoftranscript|>"
    buffer: str = ""
    num_of_words_history: List[str] = field(default_factory=list)
    lang: str | None = None
    lang_token: str | None = None
    

@dataclass
class ListenerContext:
    session_id: str
    redis_keymap: SessionKeyMap
    stream_state: Dict = field(default_factory=dict)

