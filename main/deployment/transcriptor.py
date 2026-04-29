import os
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

from prepare import pick_device, XVectorExtractor

import re

# from patch import gpu_model_runner

@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=24,
    ray_actor_options={
        "num_gpus": 0.2,
        "runtime_env": {
            "env_vars": {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "VLLM_KNN_OUT_DIR": "/data/kissos_SY_test/output",
                "VLLM_KNN_RESCORE_BATCH_SIZE": "128",
                "VLLM_KNN_ASR_TOPM": "8",
                
            },
        },
    },
)

class DeployWhisper:
    def __init__(self):

        # KNN 모델과, datastore가 있는 경로-추후에 수정해야함
        os.environ["VLLM_KNN_OUT_DIR"] = "/data/kissos_SY_test/output"

        # 한번에 최대 배치 처리 가능 갯수
        os.environ["VLLM_KNN_RESCORE_BATCH_SIZE"] = "128"
        os.environ["VLLM_KNN_ASR_TOPM"] = "8"

        # 모델정보 불러오기
        with open('whisper_config.yaml', 'r', encoding='utf-8') as f:
            kwargs = yaml.safe_load(f)

        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**kwargs))
        self.session_pool = {}
        self.r = redis.Redis(host='localhost', port=6380, db=0, decode_responses=True)

        # KNN시 필요한 X-vector 추출
        self.device = pick_device("auto")
        self.xvec_extractor = XVectorExtractor(self.device)

    def reconfigure(self, new_config):
        # 새로운 설정으로 엔진을 재구성하는 로직을 구현합니다.
        # 예를 들어, 새로운 모델로 교체하거나, GPU 메모리 활용도를 변경하는 등의 작업이 있을 수 있습니다.
        # new_args = AsyncEngineArgs(**new_config)
        # self.engine = AsyncLLMEngine.from_engine_args(new_args)
        pass



    async def generate(
            self, 
            encoder_input, 
            decoder_input, 
            allowed_languages,
            # redis_keys,
            ) -> AsyncGenerator[str, None]:

        
        request_id = random_uuid()
        prompt, sampling_params = self.preprocess_request(
            encoder_input, 
            decoder_input, 
            allowed_languages
            )
        print(f"[transcription generate called]\nlanguages: {allowed_languages}\nrequest_id: {request_id}\n")
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        output = None
        async for request_output in results_generator:
            output = request_output

        # redis_keys.transcription_s = output
        return output

    def preprocess_request(
        self,
        encoder_input, 
        decoder_input, 
        allowed_languages
        ):

        # x-vector 설정
        qx = self.xvec_extractor(encoder_input).float()
        qx = qx / (qx.norm(p=2) + 1e-8)
        knn_qx = qx.detach().cpu().tolist()

        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": encoder_input,
                },
            },
            "decoder_prompt": decoder_input,
        }

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=448,
            skip_special_tokens=False,
            extra_args={
                "allowed_languages": allowed_languages,
                "knn_qx": knn_qx # KNN x-vector 추가
                }
        )
        
        # print("knn_qx len =", len(knn_qx), "first3 =", knn_qx[:3])
        # print("sampling extra_args keys =", sampling_params.extra_args.keys())

        return prompt, sampling_params
