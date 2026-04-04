## Architecture Overview

### Blueprint
![vLLM LID](./imgs/Architecture%20Mechanism.png)

---

### Server
**Nginx**는 앞단에 배치되어 웹/프록시 서버로 기능한다.
- Speaker의 WebSocket 요청은 프록시를 통과하여 서버 내부의 Ingress에 도착하고 Ingress는 내부의 WebSocket 엔드포인트에 요청을 전달하여 Socket Handler를 깨운다.  
- Listener의 HTTP 요청은 /entrance에 도착하여 Nginx의 프론트엔드와 백엔드를 깨운다. /entrance는 세션 ID를 입력받고 서버에서 해당 세션이 진행중인지를 확인한다. 만약 세션이 진행중이라면 서버 내부의 Ingress와 SSE 연결을 수립하여 SSE Handler를 깨운다. 이때 유저는 /session/{sessionID} 페이지로 리디렉션되어 전사/번역 화면으로 이동한다.

**Ray Serve**는 대규모 추론 환경을 간편하게 구성할 수 있도록 하는 프레임워크로 진입점과 워커를 비롯한 여러 독립적인 프로세스를 GIL을 우회해 통합적으로 제어하는 데 탁월하다. 프로세스는 진입점이자 세션 매니지먼트를 담당하는 Ingress Server와 모델 워커로 동작하는 DeployWhisper, 그리고 DeployTranslator(모델 미정)로 구성된다. 

- **Ingress Server**는 FastAPI를 진입점으로 사용하는 비동기 API 서버로 내부적으로 Socket Handler와 SSE Handler를 가진다.
    - **Socket Handler**는 Socket/Transcription/Translation Loop로 구성되며 각 루프의 작업은 async queue로 연결되어 완전히 비동기적으로 동작한다.    
    Socket Loop는 웹소켓의 연결상태, 클라이언트가 전송하는 오디오 데이터 등을 핸들링하며 ClientManger 객체를 이용해 클라이언트 상태를 기록하고 제어한다.   
    Transcription/Translation Loop는 Whisper/Translator Deployment에 추론 요청을 보내며 그 결과값을 받아와 후처리한다. 완성된 결과값은 문장의 완료/미완료 여부에 따라 Redis의 Steram, 혹은 Pub/Sub에 기록된다.

    - **SSE Handler**는 Nginx의 백엔드로부터 요청받는 세션 ID로 Redis의 Stream, Pub/Sub을 구독하고 결과값을 사용자에게 보낸다.

- **Deploy Whisper/Translator**는 vLLM을 사용해 개별 추론 요청을 Continues Batching으로 묶어 처리하는 모델 워커로 추론 설정, 모델 튜닝, 전처리 등을 담당한다.

---

### Speaker
WebSocket 연결을 사용하며 프론트엔드와 백엔드가 크롬 확장 프로그램 형태로 제공된다.  

- 프론트엔드는 녹음을 켜고 끌 수 있는 메인 페이지와 전사/번역 언어를 선택할 수 있는 설정 페이지, 전사/번역 결과를 볼 수 있는 보기 페이지로 이루어진다.

- 백엔드는 확장 프로그램을 케어하기 위한 popup, background, offscreen 등의 로직을 가지고 있으며 특히 offscreen에서는 웹소켓 연결 핸들링과 VAD 상태 머신 등의 핵심 로직들이 실행된다.

---

### Listener
SSE 연결을 사용하며 프론트엔드와 백엔드는 HTTP 연결 수립 후 Nginx에서 제공된다.   
아래와 같은 UI 플로우를 가진다.

![vLLM LID](./imgs/Listener%20Frontend.png)


## How to use

### Environment Setup

```bash
# with pyproject.toml
uv sync
```

```bash
# from zero
uv init --python 3.12.13
uv sync
uv add "ray[serve]==2.54.0" "vllm==0.17.0" redis pyyaml
```

### Start Redis Container

```bash
docker run -d \
  --name redis-kissos \
  -p 6380:6379 \
  -v /data/kissos/redis/data:/data \ # 볼륨 마운트는 환경에 맞게 수정하기
  --restart always \
  redis:latest redis-server --appendonly yes
```
### Launch Server (simplified)
```bash
source .venv/bin/activate
cd ./main
serve run serve_main:runner
```

### Custom vLLM logic
언어 지정 LID를 사용하고 싶다면 아래 파일을 수정할 것
- **vllm/v1/worker/gpu_model_runner.py** 

![vLLM LID](./imgs/vLLM%20Custom%20LID.png)

```python
# under line 3679
if "whisper" in self.model_config.model and scheduler_output.scheduled_new_reqs:
    self.mask_logits_for_LID(scheduler_output, logits)
```
```python
# under excute_model define
def mask_logits_for_LID(self, scheduler_output, logits) -> None:
    req_id_to_index = {req_id: i for i, req_id in enumerate(self.input_batch.req_ids)}
    target_req_indices = []
    row_indices = []
    col_indices = []

    for new_req in scheduler_output.scheduled_new_reqs:
        if new_req.prompt_token_ids[0] != 50362: # <|startofprev|>
            req_index = req_id_to_index[new_req.req_id]
            req_allowed_langs = new_req.sampling_params.extra_args["allowed_languages"]
            req_allowed_tokens = torch.tensor(req_allowed_langs, device=logits.device)

            target_req_indices.append(req_index)
            row_indices.append(torch.full_like(req_allowed_tokens, req_index))
            col_indices.append(req_allowed_tokens)

    if not target_req_indices:
        return

    row_indices = torch.cat(row_indices)
    col_indices = torch.cat(col_indices)

    allowed_logits_of_target_reqs = logits[row_indices, col_indices]
    logits[target_req_indices] = float('-inf')
    logits[row_indices, col_indices] = allowed_logits_of_target_reqs
```


## About Ray Serve

### Architecture Comparison
![vLLM LID](./imgs/Architecture%20Comparison.png)

### Key concepts of Ray Serve
되로록 아래의 내용을 숙지할 것
- Ray: https://docs.ray.io/en/latest/ray-core/key-concepts.html
- Ray Serve: https://docs.ray.io/en/latest/serve/key-concepts.html