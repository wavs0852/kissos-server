
from dataclasses import dataclass, field
import asyncio


@dataclass
class TranscriptionState:
    prompt: str = "<|startoftranscript|>"
    buffer: str = ""
    num_of_words_history: list = field(default_factory=list)
    lang_token: str | None = None