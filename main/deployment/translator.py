import uuid
import asyncio
from typing import AsyncGenerator
import redis.asyncio as redis

import ray
from ray import serve

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.sampling_params import BeamSearchParams

import yaml

# 지원 언어 리스트
from constant.qwen import COMMON_LANG_MAP

@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=24, 
    ray_actor_options={
        "num_gpus": 0.3,
        "runtime_env": {
            "env_vars": {"VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                         "CUDA_VISIBLE_DEVICES" : "2"}
            
        }
    }
)

class DeployQwen:
    def __init__(self):

        # 모델정보 불러오기
        with open('qwen2.5-14B_config.yaml', 'r', encoding='utf-8') as f:
            kwargs = yaml.safe_load(f)

        # 기존 모델 병렬처리에 Redis 추가 호
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**kwargs))
        self.session_pool = {}
        self.r = redis.Redis(host='localhost', port=6380, db=0, decode_responses=True)

        self.COMMON_LANG_MAP = COMMON_LANG_MAP
        
    def reconfigure(self, new_config):
        # 새로운 설정으로 엔진을 재구성하는 로직을 구현합니다.
        # 예를 들어, 새로운 모델로 교체하거나, GPU 메모리 활용도를 변경하는 등의 작업이 있을 수 있습니다.
        # new_args = AsyncEngineArgs(**new_config)
        # self.engine = AsyncLLMEngine.from_engine_args(new_args)
        pass


    async def generate(
        self,
        cur_lid,
        trans_langs,
        text,
    ) -> dict:


        if not trans_langs:
            return {}

        # 문장하나씩 번역하는 코드
        async def _translate_one(target_lang):
            request_id = random_uuid()
            prompt, sampling_params = self.preprocess_request(
                text=text,
                source_lang=cur_lid,
                target_lang=target_lang,
            )

            results_generator = self.engine.generate(prompt, sampling_params, request_id)

            output = None
            async for request_output in results_generator:
                output = request_output

            if output is None or not output.outputs:
                return target_lang, ""

            return target_lang, output.outputs[0].text.strip()

        # gather 실행
        tasks = [_translate_one(target_lang) for target_lang in trans_langs]
        translated_list = await asyncio.gather(*tasks, return_exceptions=True)

        # 나온 결과로 딕셔너리 생성
        result = {}
        for target_lang, item in zip(trans_langs, translated_list):
            if isinstance(item, Exception):
                result[target_lang] = ""
            else:
                lang, translated_text = item
                result[lang] = translated_text

        return result

    def preprocess_request(
            self,
            text,
            source_lang,
            target_lang,
        ):

        # 딕셔너리에서 언어 단어 가져오기
        source_lang_name = self.COMMON_LANG_MAP.get(source_lang, source_lang)
        target_lang_name = self.COMMON_LANG_MAP.get(target_lang, target_lang)

        #  prompt = f"Translate following text to {target_lang} without any description:\n\nText: {text}
        prompt = f"""You are a professional translation engine.

        Translate the text from {source_lang_name} to {target_lang_name}.

        Rules:
        - Output only the translated text.
        - Do not add explanations, notes, labels, or quotes.
        - Preserve the original meaning.
        - Preserve line breaks and formatting.
        - Do not translate proper nouns unless they have a standard translation.
        - If the input is already in {target_lang_name}, return it unchanged.

        Text:
        {text}

        Translation:""" 

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=228,
            skip_special_tokens=True,
        )

        return prompt, sampling_params
