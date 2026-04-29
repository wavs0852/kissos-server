from typing import Any, Dict
from pydantic import BaseModel, ConfigDict

class PublishForm(BaseModel):
    type: str
    payload: Dict[str, Any] = {}


class MessageForm(BaseModel):
    """
    base class for TranscriptionMessage and TranslationMessage

    type: str
    status: str
    """
    model_config = ConfigDict(extra='allow')
    type: str
    status: str

class TranscriptionMessage(MessageForm):
    """
    type: str
    status: str
    number: int
    complete: str ("true" or "false")
    result: str
    """
    number: int
    complete: str
    language: str
    result: str

class TranslationMessage(MessageForm):
    """ 
    type: str
    status: str
    number: int 
    language: str (comma-separated if multiple)
    translation_1: str (unpack from **translation result)
    translation_2: str (ConfigDict(extra='allow')를 상속받아 딕셔너리 언팩이 가능함)
    ... 
    """
    number: int
    language: str