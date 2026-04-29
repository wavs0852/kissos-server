import os
import sys
import json
import math
import time
import argparse
import difflib
import csv
import re
from collections import defaultdict, OrderedDict
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import torchaudio
except Exception:
    torchaudio = None

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except Exception:
    NearestNeighbors = None

from transformers import WhisperProcessor, WhisperForConditionalGeneration


TRANSCRIPTION_ALT_PATTERN = re.compile(r"\(([^()]*)\)\s*/\s*\(([^()]*)\)")


def _die(msg: str, code: int = 2):
    raise SystemExit(msg)


def make_tqdm(iterable, **kwargs):
    return tqdm(
        iterable,
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0.2,
        ascii=True,
        leave=True,
        disable=False,
        **kwargs,
    )


def set_torch_perf_flags():
    torch.set_num_threads(26)
    torch.set_num_interop_threads(26)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pick_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def normalize_transcription_text(text: str) -> str:
    if text is None:
        return ""

    out = str(text)

    while True:
        new_out = TRANSCRIPTION_ALT_PATTERN.sub(lambda m: m.group(2).strip(), out)
        if new_out == out:
            break
        out = new_out

    out = re.sub(r"\s+", " ", out).strip()
    return out


def extract_transcription_text(jdata: dict) -> str:
    return str(jdata.get("06_transcription", {}).get("1_text", ""))


def update_transcription_text_in_json_file(json_path: Path) -> tuple[bool, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        jdata = json.load(f)

    raw_text = str(jdata.get("06_transcription", {}).get("1_text", ""))
    normalized_text = normalize_transcription_text(raw_text)

    if "06_transcription" not in jdata or not isinstance(jdata["06_transcription"], dict):
        jdata["06_transcription"] = {}

    changed = raw_text != normalized_text
    jdata["06_transcription"]["1_text"] = normalized_text

    if changed:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(jdata, f, ensure_ascii=False, indent=2)

    return changed, normalized_text


def load_audio_mono_16k(path: Path, target_sr: int = 16000) -> np.ndarray:
    if sf is not None:
        wav, sr = sf.read(str(path), always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32, copy=False)
        if sr != target_sr:
            if torchaudio is None:
                _die("Resample 필요하지만 torchaudio 없음. torchaudio 설치하거나 16k wav로 변환하세요.")
            w = torch.from_numpy(wav).unsqueeze(0)
            w = torchaudio.functional.resample(w, sr, target_sr)
            wav = w.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        return np.ascontiguousarray(wav)

    if torchaudio is None:
        _die("soundfile/torchaudio 둘 다 없음. pip install soundfile torchaudio")

    wav, sr = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return np.ascontiguousarray(wav.squeeze(0).cpu().numpy().astype(np.float32, copy=False))


def fast_audio_duration(path: Path) -> float:
    if sf is not None:
        try:
            info = sf.info(str(path))
            return float(info.frames) / float(info.samplerate)
        except Exception:
            pass

    if torchaudio is not None:
        try:
            if hasattr(torchaudio, "info"):
                info = torchaudio.info(str(path))
                return float(info.num_frames) / float(info.sample_rate)
        except Exception:
            pass

    _die(f"duration 계산 불가: {path}")


class XVectorExtractor:
    def __init__(self, device: torch.device):
        self.device = device
        self.mode = "mfcc"
        self.model = None
        self.mfcc = None

        if torchaudio is None:
            self.mode = "none"
            return

        try:
            bundle = torchaudio.pipelines.SUPERB_XVECTOR
            self.model = bundle.get_model().to(device).eval()
            if device.type == "cuda":
                self.model = self.model.half()
            self.mode = "superb"
            return
        except Exception:
            self.model = None

        try:
            self.mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40).to(device)
            self.mode = "mfcc"
        except Exception:
            self.mode = "none"

    @torch.inference_mode()
    def __call__(self, wav_16k: np.ndarray) -> torch.Tensor:
        if self.mode == "none" or torchaudio is None:
            return torch.zeros(40, device=self.device, dtype=torch.float32)

        dtype = torch.float16 if (self.device.type == "cuda" and self.mode == "superb") else torch.float32
        w = torch.from_numpy(np.ascontiguousarray(wav_16k).copy()).to(device=self.device, dtype=dtype).unsqueeze(0)

        if self.mode == "superb" and self.model is not None:
            out = self.model(w)
            if isinstance(out, (tuple, list)):
                out = out[0]
            emb = out.squeeze(0).float()
            emb = emb / (emb.norm(p=2) + 1e-8)
            return emb

        if self.mode == "mfcc" and self.mfcc is not None:
            m = self.mfcc(w.float())
            if m.ndim == 3:
                m = m.squeeze(0)
            elif m.ndim == 4:
                m = m.squeeze(0).squeeze(0)
            emb = m.mean(dim=-1).float()
            emb = emb / (emb.norm(p=2) + 1e-8)
            return emb

        return torch.zeros(40, device=self.device, dtype=torch.float32)


def ko_lang_name(lang: str) -> str:
    m = {
        "ko": "korean",
        "kr": "korean",
        "en": "english",
        "ja": "japanese",
        "zh": "chinese",
    }
    return m.get(lang.lower().strip(), lang)


@lru_cache(maxsize=32)
def _strict_lower_mask(k: int) -> np.ndarray:
    return np.tril(np.ones((k, k), dtype=bool), k=-1)


def distinct_counts_batch(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals)
    if vals.ndim != 2:
        _die("distinct_counts_batch expects 2D array")
    bsz, k = vals.shape
    if k == 0:
        return np.zeros((bsz, 0), dtype=np.float32)
    eq = vals[:, :, None] == vals[:, None, :]
    seen_before = (eq & _strict_lower_mask(k)[None, :, :]).any(axis=2)
    first = ~seen_before
    return np.cumsum(first, axis=1, dtype=np.int32).astype(np.float32)


@torch.inference_mode()
def distinct_counts_batch_torch(vals: torch.Tensor):
    bsz, k = vals.shape
    if k == 0:
        return torch.zeros((bsz, 0), device=vals.device, dtype=torch.float32)
    lower = torch.tril(torch.ones((k, k), device=vals.device, dtype=torch.bool), diagonal=-1)
    eq = vals[:, :, None].eq(vals[:, None, :])
    seen_before = (eq & lower.unsqueeze(0)).any(dim=2)
    first = (~seen_before).to(torch.float32)
    return torch.cumsum(first, dim=1)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_whisper(model_id: str, device: torch.device, dtype: torch.dtype):
    processor = WhisperProcessor.from_pretrained(model_id)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(device).eval()
    except Exception:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device).eval()
    return processor, model


@torch.inference_mode()
def whisper_encode_batch(model, processor, wavs_16k, device: torch.device, dtype: torch.dtype):
    feats = processor.feature_extractor(
        wavs_16k,
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features
    feats = feats.to(device=device, dtype=dtype, non_blocking=True)
    return model.model.encoder(feats, return_dict=True)


def get_decoder_preset(model, processor, device: torch.device, language: str):
    if language.lower() in ["auto", ""]:
        prompt = processor.get_decoder_prompt_ids(task="transcribe")
    else:
        lang = ko_lang_name(language)
        prompt = processor.get_decoder_prompt_ids(language=lang, task="transcribe")

    if len(prompt) > 0 and isinstance(prompt[0], (tuple, list)):
        prompt_ids = [int(x[1]) for x in prompt]
    else:
        prompt_ids = [int(x) for x in prompt]
    start_id = model.config.decoder_start_token_id
    eos = processor.tokenizer.eos_token_id
    prefix = torch.tensor([[start_id] + prompt_ids], device=device, dtype=torch.long)
    return prefix, eos, processor.tokenizer


def build_30s_chunks(wav_16k: np.ndarray, sr: int = 16000, sec: int = 30):
    chunk_len = sec * sr
    total_len = len(wav_16k)
    chunks = []
    for start in range(0, total_len, chunk_len):
        end = min(start + chunk_len, total_len)
        chunk = wav_16k[start:end]
        if len(chunk) < chunk_len:
            pad = np.zeros(chunk_len - len(chunk), dtype=np.float32)
            chunk = np.concatenate((chunk, pad), axis=0)
        chunks.append(np.ascontiguousarray(chunk, dtype=np.float32))
    return chunks


@torch.inference_mode()
def greedy_generate_collect_batch(
    model,
    enc,
    prefix: torch.Tensor,
    eos: int,
    max_new_tokens: int,
):
    device = prefix.device
    bsz = enc.last_hidden_state.shape[0]
    dec0 = prefix.repeat(bsz, 1)

    out = model(
        encoder_outputs=enc,
        decoder_input_ids=dec0,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past = out.past_key_values

    finished = torch.zeros(bsz, device=device, dtype=torch.bool)
    toks = []
    logits_steps = []
    hidden_steps = []

    for _ in range(max_new_tokens):
        logits_step = out.logits[:, -1, :].float()
        hidden_step = out.decoder_hidden_states[-1][:, -1, :].float()

        nxt = torch.argmax(logits_step, dim=-1)
        nxt = torch.where(finished, torch.full_like(nxt, eos), nxt)

        toks.append(nxt)
        logits_steps.append(logits_step)
        hidden_steps.append(hidden_step)

        finished = finished | nxt.eq(eos)
        if bool(finished.all()):
            break

        out = model(
            encoder_outputs=enc,
            decoder_input_ids=nxt.unsqueeze(1),
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = out.past_key_values

    if len(toks) == 0:
        empty_tok = torch.empty((bsz, 0), device=device, dtype=torch.long)
        empty_h = torch.empty((bsz, 0, model.config.d_model), device=device, dtype=torch.float32)
        empty_logits = torch.empty((bsz, 0, model.config.vocab_size), device=device, dtype=torch.float32)
        lens = torch.zeros((bsz,), device=device, dtype=torch.long)
        return empty_tok, empty_h, empty_logits, lens

    toks = torch.stack(toks, dim=1)
    hidden = torch.stack(hidden_steps, dim=1)
    logits = torch.stack(logits_steps, dim=1)

    is_eos = toks.eq(eos)
    has_eos = is_eos.any(dim=1)
    first_eos = is_eos.float().argmax(dim=1).long()
    lens = torch.where(has_eos, first_eos, torch.full_like(first_eos, toks.shape[1]))
    return toks, hidden, logits, lens


@torch.inference_mode()
def flatten_generation_for_rescore(
    base_tokens: torch.Tensor,
    hidden: torch.Tensor,
    logits: torch.Tensor,
    lens: torch.Tensor,
):
    if base_tokens.ndim != 2 or hidden.ndim != 3 or logits.ndim != 3:
        _die("invalid generation tensors")

    bsz = base_tokens.shape[0]
    if bsz == 0:
        return None, None, lens, None

    h_list = []
    logits_list = []
    owner_list = []

    for i in range(bsz):
        li = int(lens[i].item())
        if li <= 0:
            continue
        h_list.append(hidden[i, :li, :])
        logits_list.append(logits[i, :li, :])
        owner_list.append(torch.full((li,), i, device=base_tokens.device, dtype=torch.long))

    if len(h_list) == 0:
        return None, None, lens, None

    h_flat = torch.cat(h_list, dim=0).contiguous()
    logits_flat = torch.cat(logits_list, dim=0).contiguous()
    owners = torch.cat(owner_list, dim=0).contiguous()
    return h_flat, logits_flat, lens, owners


class SmallArrayCache:
    def __init__(self, max_items: int = 8):
        self.max_items = max_items
        self.od = OrderedDict()

    def get(self, key, loader):
        if key in self.od:
            v = self.od.pop(key)
            self.od[key] = v
            return v
        v = loader()
        self.od[key] = v
        while len(self.od) > self.max_items:
            self.od.popitem(last=False)
        return v


class ShardedIntLookup:
    def __init__(self, items, field, cache_size=256):
        self.items = sorted(items, key=lambda x: x["start"])
        self.field = field
        self.starts = np.asarray([int(x["start"]) for x in self.items], dtype=np.int64)
        self.ends = np.asarray([int(x["end"]) for x in self.items], dtype=np.int64)
        self.cache = SmallArrayCache(cache_size)

    def _find_item_index(self, idx):
        pos = int(np.searchsorted(self.ends, idx, side="right"))
        if pos >= len(self.items):
            _die(f"global index out of range: {idx}")
        item = self.items[pos]
        if not (int(item["start"]) <= idx < int(item["end"])):
            _die(f"index-shard mapping failed: {idx}")
        return pos

    def _load_arr(self, path: str):
        return np.load(path, mmap_mode="r")

    def lookup(self, ids: np.ndarray, out_dtype):
        ids = np.asarray(ids, dtype=np.int64)
        flat = ids.reshape(-1)
        out = np.empty(flat.shape[0], dtype=out_dtype)
        if flat.size == 0:
            return out.reshape(ids.shape)

        shard_pos = np.array([self._find_item_index(int(i)) for i in flat], dtype=np.int64)
        for sp in np.unique(shard_pos):
            mask = shard_pos == sp
            item = self.items[int(sp)]
            arr = self.cache.get(item[self.field], lambda p=item[self.field]: self._load_arr(p))
            local = flat[mask] - int(item["start"])
            out[mask] = np.asarray(arr[local], dtype=out_dtype)
        return out.reshape(ids.shape)


class ShardedXvecStore:
    def __init__(self, items, cache_size=256):
        self.items = sorted(items, key=lambda x: x["start"])
        self.ends = np.asarray([int(x["end"]) for x in self.items], dtype=np.int64)
        self.cache = SmallArrayCache(cache_size)

    def _find_item_index(self, idx):
        pos = int(np.searchsorted(self.ends, idx, side="right"))
        if pos >= len(self.items):
            _die(f"utt_id out of range: {idx}")
        item = self.items[pos]
        if not (int(item["start"]) <= idx < int(item["end"])):
            _die(f"utt shard mapping failed: {idx}")
        return pos

    def _load_arr(self, path: str):
        return np.load(path, mmap_mode="r")

    def lookup_batch(self, utt_ids: np.ndarray) -> np.ndarray:
        utt_ids = np.asarray(utt_ids, dtype=np.int64)
        flat = utt_ids.reshape(-1)
        if flat.size == 0:
            return np.empty((0, 0), dtype=np.float32)

        out_dim = None
        out = None
        shard_pos = np.array([self._find_item_index(int(i)) for i in flat], dtype=np.int64)

        for sp in np.unique(shard_pos):
            mask = shard_pos == sp
            item = self.items[int(sp)]
            arr = self.cache.get(item["path"], lambda p=item["path"]: self._load_arr(p))
            if out is None:
                out_dim = int(arr.shape[1])
                out = np.empty((flat.shape[0], out_dim), dtype=np.float32)
            local = flat[mask] - int(item["start"])
            out[mask] = np.asarray(arr[local], dtype=np.float32)

        return out.reshape(*utt_ids.shape, out_dim)


class TrainShardSampler:
    def __init__(self, manifest, cache_size=256):
        self.items = sorted(manifest["train_shards"], key=lambda x: x["id"])
        if len(self.items) == 0:
            _die("train shard가 없습니다.")
        self.counts = np.asarray([int(x["n"]) for x in self.items], dtype=np.int64)
        self.cum = np.cumsum(self.counts)
        self.total = int(self.cum[-1])
        self.cache = SmallArrayCache(cache_size)

    def _load_item(self, item):
        return {
            "key": np.load(item["key"], mmap_mode="r"),
            "tgt": np.load(item["tgt"], mmap_mode="r"),
            "logit_t": np.load(item["logit_t"], mmap_mode="r"),
            "logsumexp": np.load(item["logsumexp"], mmap_mode="r"),
            "utt_id": np.load(item["utt_id"], mmap_mode="r"),
        }

    def sample_batch(self, bs: int):
        if bs <= 0:
            _die("batch size must be > 0")
        global_idx = np.random.randint(0, self.total, size=(bs,), dtype=np.int64)
        shard_pos = np.searchsorted(self.cum, global_idx, side="right")
        local_idx = global_idx - np.where(shard_pos > 0, self.cum[shard_pos - 1], 0)

        q_key = []
        q_tgt = []
        q_logit_t = []
        q_lse = []
        q_utt = []

        for sp in np.unique(shard_pos):
            mask = shard_pos == sp
            item = self.items[int(sp)]
            arrs = self.cache.get(item["key"], lambda it=item: self._load_item(it))
            loc = local_idx[mask]
            q_key.append(np.asarray(arrs["key"][loc], dtype=np.float32))
            q_tgt.append(np.asarray(arrs["tgt"][loc], dtype=np.int64))
            q_logit_t.append(np.asarray(arrs["logit_t"][loc], dtype=np.float32))
            q_lse.append(np.asarray(arrs["logsumexp"][loc], dtype=np.float32))
            q_utt.append(np.asarray(arrs["utt_id"][loc], dtype=np.int64))

        key = np.concatenate(q_key, axis=0)
        tgt = np.concatenate(q_tgt, axis=0)
        logit_t = np.concatenate(q_logit_t, axis=0)
        lse = np.concatenate(q_lse, axis=0)
        utt = np.concatenate(q_utt, axis=0)

        perm = np.random.permutation(bs)
        return key[perm], tgt[perm], logit_t[perm], lse[perm], utt[perm]


class ShardedDatastore:
    def __init__(
        self,
        out_dir: Path,
        prefer_faiss: bool = True,
        cache_size=256,
        ivf_nlist: int | None = None,
        ivf_nprobe: int | None = None,
        pq_m: int | None = None,
        pq_nbits: int = 8,
        faiss_train_samples: int | None = None,
        use_faiss_gpu: bool | None = None,
    ):
        self.out_dir = out_dir
        self.manifest = load_json(out_dir / "manifest.json")
        self.key_shards = sorted(self.manifest["datastore_shards"], key=lambda x: x["start"])
        self.n_keys = int(self.manifest["n_keys"])
        self.dim = int(self.manifest["keys_dim"])
        self.use_faiss = bool(prefer_faiss and (faiss is not None))
        self.index = None
        self.index_cpu = None
        self.nn = None
        self.total_search_time = 0.0
        self.search_backend = "none"
        self.index_type = "none"
        self.key_cache = SmallArrayCache(cache_size)
        self.val_lookup = ShardedIntLookup(self.key_shards, "vals", cache_size=cache_size)
        self.utt_lookup = ShardedIntLookup(self.key_shards, "utt_ids", cache_size=cache_size)
        self.xvec_store = ShardedXvecStore(self.manifest["utt_xvec_shards"], cache_size=256)

        cfg = self.manifest.get("search_config", {})

        raw_ivf_nlist = ivf_nlist if ivf_nlist is not None else cfg.get("ivf_nlist")
        if raw_ivf_nlist is None:
            raw_ivf_nlist = self._auto_nlist()
        self.ivf_nlist = int(raw_ivf_nlist)

        raw_ivf_nprobe = ivf_nprobe if ivf_nprobe is not None else cfg.get("ivf_nprobe")
        if raw_ivf_nprobe is None:
            raw_ivf_nprobe = self._auto_nprobe(self.ivf_nlist)
        self.ivf_nprobe = int(raw_ivf_nprobe)

        raw_pq_m = pq_m if pq_m is not None else cfg.get("pq_m")
        if raw_pq_m is None:
            raw_pq_m = self._auto_pq_m()
        self.pq_m = int(raw_pq_m)

        raw_pq_nbits = pq_nbits if pq_nbits is not None else cfg.get("pq_nbits")
        if raw_pq_nbits is None:
            raw_pq_nbits = 8
        self.pq_nbits = int(raw_pq_nbits)

        raw_faiss_train_samples = faiss_train_samples if faiss_train_samples is not None else cfg.get("faiss_train_samples")
        if raw_faiss_train_samples is None:
            raw_faiss_train_samples = min(max(self.n_keys, 1), 200000)
        self.faiss_train_samples = int(raw_faiss_train_samples)

        gpu_default = cfg.get("use_faiss_gpu", True)
        self.use_faiss_gpu = bool(gpu_default if use_faiss_gpu is None else use_faiss_gpu)
        self.gpu_resources = None

    def _load_keys(self, path: str):
        return np.load(path, mmap_mode="r")

    def _auto_nlist(self) -> int:
        if self.n_keys <= 0:
            return 256
        raw = int(round(math.sqrt(self.n_keys) * 4.0))
        return max(256, min(8192, raw))

    def _auto_nprobe(self, nlist: int) -> int:
        return max(8, min(64, max(1, nlist // 64)))

    def _auto_pq_m(self) -> int:
        candidates = [64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1]
        for m in candidates:
            if m <= self.dim and self.dim % m == 0:
                return m
        return 1

    def _has_faiss_gpu(self) -> bool:
        return bool(
            faiss is not None
            and hasattr(faiss, "StandardGpuResources")
            and hasattr(faiss, "index_cpu_to_gpu")
            and self.use_faiss_gpu
            and torch.cuda.is_available()
        )

    def _to_gpu_if_possible(self):
        if self.index_cpu is None:
            return
        if not self._has_faiss_gpu():
            self.index = self.index_cpu
            return
        try:
            self.gpu_resources = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index_cpu)
            self.search_backend = "faiss_gpu"
            self._set_runtime_search_params(self.index)
        except Exception:
            self.index = self.index_cpu

    def _set_runtime_search_params(self, index_obj):
        if index_obj is None:
            return
        try:
            if hasattr(index_obj, "nprobe"):
                index_obj.nprobe = int(self.ivf_nprobe)
        except Exception:
            pass
        try:
            if hasattr(index_obj, "quantizer_efSearch"):
                index_obj.quantizer_efSearch = max(32, int(self.ivf_nprobe) * 2)
        except Exception:
            pass

    def _sample_training_keys(self, max_samples: int) -> np.ndarray:
        max_samples = int(max(1, max_samples))
        total = 0
        pieces = []
        for shard in self.key_shards:
            arr = self.key_cache.get(shard["keys"], lambda p=shard["keys"]: self._load_keys(p))
            n = int(arr.shape[0])
            if n <= 0:
                continue
            remaining = max_samples - total
            if remaining <= 0:
                break
            take = min(n, max(2048, remaining // max(1, len(self.key_shards) - len(pieces))))
            if take < n:
                step = max(1, n // take)
                sample = np.asarray(arr[::step][:take], dtype=np.float32)
            else:
                sample = np.asarray(arr, dtype=np.float32)
            pieces.append(np.ascontiguousarray(sample, dtype=np.float32))
            total += int(sample.shape[0])
            if total >= max_samples:
                break
        if len(pieces) == 0:
            _die("FAISS train sample 생성 실패")
        out = np.concatenate(pieces, axis=0)
        if out.shape[0] > max_samples:
            idx = np.linspace(0, out.shape[0] - 1, num=max_samples, dtype=np.int64)
            out = out[idx]
        return np.ascontiguousarray(out, dtype=np.float32)

    def _build_ivfpq_index(self):
        if not hasattr(faiss, "IndexIVFPQ"):
            return None, None
        nlist = max(1, min(int(self.ivf_nlist), max(1, self.n_keys // 32)))
        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, int(self.pq_m), int(self.pq_nbits), faiss.METRIC_L2)
        train_x = self._sample_training_keys(self.faiss_train_samples)
        index.train(train_x)
        for shard in make_tqdm(self.key_shards, desc="FAISS IVFPQ build"):
            keys = np.asarray(self.key_cache.get(shard["keys"], lambda p=shard["keys"]: self._load_keys(p)), dtype=np.float32)
            ids64 = np.arange(int(shard["start"]), int(shard["end"]), dtype=np.int64)
            index.add_with_ids(keys, ids64)
        return index, f"faiss_ivfpq_nlist{nlist}_m{self.pq_m}_nbits{self.pq_nbits}"

    def _build_ivfsq_index(self):
        if not hasattr(faiss, "IndexIVFScalarQuantizer") or not hasattr(faiss, "ScalarQuantizer"):
            return None, None
        nlist = max(1, min(int(self.ivf_nlist), max(1, self.n_keys // 32)))
        quantizer = faiss.IndexFlatL2(self.dim)
        qtype = faiss.ScalarQuantizer.QT_fp16
        index = faiss.IndexIVFScalarQuantizer(quantizer, self.dim, nlist, qtype, faiss.METRIC_L2)
        train_x = self._sample_training_keys(self.faiss_train_samples)
        index.train(train_x)
        for shard in make_tqdm(self.key_shards, desc="FAISS IVFSQ build"):
            keys = np.asarray(self.key_cache.get(shard["keys"], lambda p=shard["keys"]: self._load_keys(p)), dtype=np.float32)
            ids64 = np.arange(int(shard["start"]), int(shard["end"]), dtype=np.int64)
            index.add_with_ids(keys, ids64)
        return index, f"faiss_ivfsqfp16_nlist{nlist}"

    def _build_flat_index(self):
        base = faiss.IndexFlatL2(self.dim)
        idx = faiss.IndexIDMap2(base)
        for shard in make_tqdm(self.key_shards, desc="FAISS flat build"):
            keys = np.asarray(self.key_cache.get(shard["keys"], lambda p=shard["keys"]: self._load_keys(p)), dtype=np.float32)
            ids64 = np.arange(int(shard["start"]), int(shard["end"]), dtype=np.int64)
            idx.add_with_ids(keys, ids64)
        return idx, "faiss_flatl2_idmap"

    def build_or_load_index(self):
        index_path = self.out_dir / "faiss.index"
        meta = self.manifest

        if self.use_faiss and index_path.exists():
            self.index_cpu = faiss.read_index(str(index_path))
            self.index_type = meta.get("index_type", "faiss_unknown")
            self.search_backend = "faiss_cpu"
            self._set_runtime_search_params(self.index_cpu)
            self._to_gpu_if_possible()
            return

        if self.use_faiss:
            built = False
            builders = [
                self._build_ivfpq_index,
                self._build_ivfsq_index,
                self._build_flat_index,
            ]

            for builder in builders:
                try:
                    idx, idx_type = builder()
                    if idx is None:
                        continue
                    self.index_cpu = idx
                    self.index_type = idx_type
                    self.search_backend = "faiss_cpu"
                    self._set_runtime_search_params(self.index_cpu)
                    faiss.write_index(self.index_cpu, str(index_path))
                    built = True
                    break
                except Exception as e:
                    print(f"[FAISS fallback] {builder.__name__} 실패: {e}")
                    self.index_cpu = None
                    built = False

            if built:
                self.manifest["index_type"] = self.index_type
                self.manifest.setdefault("search_config", {})
                self.manifest["search_config"].update({
                    "ivf_nlist": int(self.ivf_nlist),
                    "ivf_nprobe": int(self.ivf_nprobe),
                    "pq_m": int(self.pq_m),
                    "pq_nbits": int(self.pq_nbits),
                    "faiss_train_samples": int(self.faiss_train_samples),
                    "use_faiss_gpu": bool(self.use_faiss_gpu),
                })
                save_json(self.out_dir / "manifest.json", self.manifest)
                self._to_gpu_if_possible()
                return

        self.search_backend = "streaming_bruteforce_cpu"
        self.index_type = "streaming_bruteforce"

    def lookup_vals(self, ids: np.ndarray) -> np.ndarray:
        return self.val_lookup.lookup(ids, np.int32)

    def lookup_utt_ids(self, ids: np.ndarray) -> np.ndarray:
        return self.utt_lookup.lookup(ids, np.int32)

    def lookup_xvecs(self, utt_ids: np.ndarray) -> np.ndarray:
        return self.xvec_store.lookup_batch(utt_ids)

    def _streaming_bruteforce_search(self, q: np.ndarray, topk: int, block_size: int = 8192):
        q = np.ascontiguousarray(q, dtype=np.float32)
        bsz = q.shape[0]

        best_d2 = np.full((bsz, topk), np.inf, dtype=np.float32)
        best_ids = np.full((bsz, topk), -1, dtype=np.int64)
        q_norm2 = np.sum(q * q, axis=1, keepdims=True).astype(np.float32)

        for shard in self.key_shards:
            keys_m = self.key_cache.get(shard["keys"], lambda p=shard["keys"]: self._load_keys(p))
            shard_start = int(shard["start"])
            shard_n = int(shard["end"]) - int(shard["start"])

            for st in range(0, shard_n, block_size):
                ed = min(st + block_size, shard_n)
                kb = np.asarray(keys_m[st:ed], dtype=np.float32)
                kb_norm2 = np.sum(kb * kb, axis=1, keepdims=False).astype(np.float32)
                d2 = q_norm2 + kb_norm2[None, :] - 2.0 * (q @ kb.T)
                block_ids = np.arange(shard_start + st, shard_start + ed, dtype=np.int64)[None, :].repeat(bsz, axis=0)

                merged_d2 = np.concatenate([best_d2, d2], axis=1)
                merged_ids = np.concatenate([best_ids, block_ids], axis=1)
                sel = np.argpartition(merged_d2, kth=topk - 1, axis=1)[:, :topk]

                row = np.arange(bsz)[:, None]
                best_d2 = merged_d2[row, sel]
                best_ids = merged_ids[row, sel]

                ord2 = np.argsort(best_d2, axis=1)
                best_d2 = best_d2[row, ord2]
                best_ids = best_ids[row, ord2]

        return best_d2.astype(np.float32, copy=False), best_ids.astype(np.int64, copy=False)

    def search_batch(self, q_f32: np.ndarray, topk: int):
        q = np.asarray(q_f32, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        q = np.ascontiguousarray(q, dtype=np.float32)

        t0 = time.perf_counter()
        if self.index is not None:
            d2, ids = self.index.search(q, topk)
            self.total_search_time += (time.perf_counter() - t0)
            return d2.astype(np.float32, copy=False), ids.astype(np.int64, copy=False)

        d2, ids = self._streaming_bruteforce_search(q, topk)
        self.total_search_time += (time.perf_counter() - t0)
        return d2, ids

    @torch.inference_mode()
    def search_batch_torch(self, q_t: torch.Tensor, topk: int):
        d2_np, ids_np = self.search_batch(q_t.detach().float().cpu().numpy(), topk)
        d2 = torch.from_numpy(np.ascontiguousarray(d2_np)).to(device=q_t.device, dtype=torch.float32, non_blocking=True)
        ids = torch.from_numpy(np.ascontiguousarray(ids_np)).to(device=q_t.device, dtype=torch.long, non_blocking=True)
        return d2, ids


class SpeakerSmoothedModule(torch.nn.Module):
    def __init__(self, K: int, hidden: int = 32, init_T: float = 1000.0, init_lambda_asr: float = 0.6):
        super().__init__()
        self.K = K
        self.t_est = torch.nn.Linear(2 * K, 1)
        self.l1 = torch.nn.Linear(2 * K, hidden)
        self.l2 = torch.nn.Linear(hidden, 1)

        with torch.no_grad():
            self.t_est.weight.zero_()
            self.t_est.bias.fill_(math.log(max(init_T, 1e-6)))
            self.l1.weight.zero_()
            self.l1.bias.zero_()
            self.l2.weight.zero_()
            self.l2.bias.fill_(math.log(init_lambda_asr / (1.0 - init_lambda_asr + 1e-8)))

    def forward(self, d_feat: torch.Tensor, c_feat: torch.Tensor, s_feat: torch.Tensor):
        ts = torch.exp(self.t_est(torch.cat([d_feat, s_feat], dim=-1))).squeeze(-1)
        ts = torch.clamp(ts, 1e-3, 1e6)

        h = torch.relu(self.l1(torch.cat([d_feat, c_feat], dim=-1)))
        lam = torch.sigmoid(self.l2(h)).squeeze(-1)
        lam = torch.clamp(lam, 1e-4, 1.0 - 1e-4)
        return ts, lam


def calc_cer(ref: str, hyp: str) -> float:
    r = ref.replace(" ", "")
    h = hyp.replace(" ", "")
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0

    if len(r) < len(h):
        r, h = h, r
    if len(h) == 0:
        return float(len(r))

    prev_row = range(len(h) + 1)
    for i, c1 in enumerate(r):
        curr_row = [i + 1]
        for j, c2 in enumerate(h):
            ins = prev_row[j + 1] + 1
            dele = curr_row[j] + 1
            subs = prev_row[j] + (c1 != c2)
            curr_row.append(min(ins, dele, subs))
        prev_row = curr_row

    distance = prev_row[-1]
    denom = len(ref.replace(" ", ""))
    return float(distance) / denom if denom > 0 else 0.0


class PrepareShardWriter:
    def __init__(self, out_dir: Path, shard_token_limit: int, utt_xvec_shard_size: int):
        self.out_dir = out_dir
        self.ds_dir = out_dir / "datastore_shards"
        self.tr_dir = out_dir / "train_shards"
        self.ux_dir = out_dir / "utt_xvec_shards"
        self.ds_dir.mkdir(parents=True, exist_ok=True)
        self.tr_dir.mkdir(parents=True, exist_ok=True)
        self.ux_dir.mkdir(parents=True, exist_ok=True)

        self.shard_token_limit = int(shard_token_limit)
        self.utt_xvec_shard_size = int(utt_xvec_shard_size)

        self.ds_buffers = {"keys": [], "vals": [], "utt_ids": []}
        self.tr_buffers = {"key": [], "tgt": [], "logit_t": [], "logsumexp": [], "utt_id": []}
        self.ux_buffer = []

        self.ds_n = 0
        self.tr_n = 0
        self.ux_n = 0

        self.global_token_start = 0
        self.global_utt_start = 0

        self.ds_shards = []
        self.tr_shards = []
        self.ux_shards = []

        self.ds_id = 0
        self.tr_id = 0
        self.ux_id = 0

    def add_tokens(self, keys, vals, utt_ids, logit_t, logsumexp):
        if keys.shape[0] != vals.shape[0] or vals.shape[0] != utt_ids.shape[0]:
            _die("token array size mismatch")
        n = int(keys.shape[0])
        if n == 0:
            return

        self.ds_buffers["keys"].append(np.ascontiguousarray(keys.astype(np.float16, copy=False)))
        self.ds_buffers["vals"].append(np.ascontiguousarray(vals.astype(np.int32, copy=False)))
        self.ds_buffers["utt_ids"].append(np.ascontiguousarray(utt_ids.astype(np.int32, copy=False)))

        self.tr_buffers["key"].append(np.ascontiguousarray(keys.astype(np.float16, copy=False)))
        self.tr_buffers["tgt"].append(np.ascontiguousarray(vals.astype(np.int32, copy=False)))
        self.tr_buffers["logit_t"].append(np.ascontiguousarray(logit_t.astype(np.float32, copy=False)))
        self.tr_buffers["logsumexp"].append(np.ascontiguousarray(logsumexp.astype(np.float32, copy=False)))
        self.tr_buffers["utt_id"].append(np.ascontiguousarray(utt_ids.astype(np.int32, copy=False)))

        self.ds_n += n
        self.tr_n += n

        if self.ds_n >= self.shard_token_limit:
            self.flush_token_shard()

    def add_utt_xvec(self, xvec):
        self.ux_buffer.append(np.ascontiguousarray(xvec.astype(np.float32, copy=False)))
        self.ux_n += 1
        if self.ux_n >= self.utt_xvec_shard_size:
            self.flush_utt_xvec_shard()

    def flush_token_shard(self):
        if self.ds_n == 0:
            return

        ds_keys = np.concatenate(self.ds_buffers["keys"], axis=0)
        ds_vals = np.concatenate(self.ds_buffers["vals"], axis=0)
        ds_utt_ids = np.concatenate(self.ds_buffers["utt_ids"], axis=0)

        tr_key = np.concatenate(self.tr_buffers["key"], axis=0)
        tr_tgt = np.concatenate(self.tr_buffers["tgt"], axis=0)
        tr_logit_t = np.concatenate(self.tr_buffers["logit_t"], axis=0)
        tr_logsumexp = np.concatenate(self.tr_buffers["logsumexp"], axis=0)
        tr_utt_id = np.concatenate(self.tr_buffers["utt_id"], axis=0)

        n = int(ds_keys.shape[0])
        ds_id = self.ds_id
        tr_id = self.tr_id

        ds_key_path = self.ds_dir / f"keys_{ds_id:06d}.npy"
        ds_val_path = self.ds_dir / f"vals_{ds_id:06d}.npy"
        ds_utt_path = self.ds_dir / f"utt_ids_{ds_id:06d}.npy"

        tr_key_path = self.tr_dir / f"key_{tr_id:06d}.npy"
        tr_tgt_path = self.tr_dir / f"tgt_{tr_id:06d}.npy"
        tr_logit_path = self.tr_dir / f"logit_t_{tr_id:06d}.npy"
        tr_lse_path = self.tr_dir / f"logsumexp_{tr_id:06d}.npy"
        tr_uttid_path = self.tr_dir / f"utt_id_{tr_id:06d}.npy"

        np.save(ds_key_path, ds_keys)
        np.save(ds_val_path, ds_vals)
        np.save(ds_utt_path, ds_utt_ids)

        np.save(tr_key_path, tr_key)
        np.save(tr_tgt_path, tr_tgt)
        np.save(tr_logit_path, tr_logit_t)
        np.save(tr_lse_path, tr_logsumexp)
        np.save(tr_uttid_path, tr_utt_id)

        self.ds_shards.append({
            "id": ds_id,
            "start": self.global_token_start,
            "end": self.global_token_start + n,
            "keys": str(ds_key_path),
            "vals": str(ds_val_path),
            "utt_ids": str(ds_utt_path),
        })
        self.tr_shards.append({
            "id": tr_id,
            "n": n,
            "key": str(tr_key_path),
            "tgt": str(tr_tgt_path),
            "logit_t": str(tr_logit_path),
            "logsumexp": str(tr_lse_path),
            "utt_id": str(tr_uttid_path),
        })

        self.global_token_start += n
        self.ds_id += 1
        self.tr_id += 1

        self.ds_buffers = {"keys": [], "vals": [], "utt_ids": []}
        self.tr_buffers = {"key": [], "tgt": [], "logit_t": [], "logsumexp": [], "utt_id": []}
        self.ds_n = 0
        self.tr_n = 0

    def flush_utt_xvec_shard(self):
        if self.ux_n == 0:
            return
        arr = np.stack(self.ux_buffer, axis=0).astype(np.float32, copy=False)
        n = int(arr.shape[0])
        ux_id = self.ux_id
        ux_path = self.ux_dir / f"utt_xvec_{ux_id:06d}.npy"
        np.save(ux_path, arr)

        self.ux_shards.append({
            "id": ux_id,
            "start": self.global_utt_start,
            "end": self.global_utt_start + n,
            "path": str(ux_path),
        })

        self.global_utt_start += n
        self.ux_id += 1
        self.ux_buffer = []
        self.ux_n = 0

    def finalize(self):
        self.flush_token_shard()
        self.flush_utt_xvec_shard()

    def build_manifest(self, model_id, language, topk, keys_dim, xvec_dim, note, index_type="none", search_config=None):
        manifest = {
            "model_id": model_id,
            "language": language,
            "K": int(topk),
            "keys_dim": int(keys_dim),
            "xvec_dim": int(xvec_dim),
            "n_keys": int(self.global_token_start),
            "n_utts": int(self.global_utt_start),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": note,
            "torch_compile": False,
            "index_type": index_type,
            "search_config": search_config or {},
            "datastore_shards": self.ds_shards,
            "train_shards": self.tr_shards,
            "utt_xvec_shards": self.ux_shards,
        }
        save_json(self.out_dir / "manifest.json", manifest)
        return manifest


def _load_audio_and_text(p):
    wav = load_audio_mono_16k(p)
    
    txt = p.stem.strip()
    
    return wav, txt


def scan_dataset_manifest(data_dir: Path, out_dir: Path):
    scan_path = out_dir / "scan_manifest.jsonl"
    total_files = 0
    ok_files = 0
    fail_files = 0

    with open(scan_path, "w", encoding="utf-8") as wf:
        for wav_path in make_tqdm(data_dir.rglob("*.wav"), desc="[1/3] scan wav"):
            total_files += 1
            try:
                dur = fast_audio_duration(wav_path)

                row = {
                    "path": str(wav_path),
                    "speaker": "spk1",
                    "train_eligible": bool(dur <= 30.0),
                    "duration": float(dur),
                }
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                ok_files += 1

            except Exception as e:
                fail_files += 1
                print(f"[scan skip] {wav_path} -> {type(e).__name__}: {e}")

    speaker_train_quota = {"spk1": max(1, int(ok_files * 0.8))}
    save_json(out_dir / "speaker_train_quota.json", speaker_train_quota)

    print(f"[*] scan total: {total_files}")
    print(f"[*] scan ok   : {ok_files}")
    print(f"[*] scan fail : {fail_files}")

    return scan_path, speaker_train_quota, ok_files

def cmd_prepare(args):
    set_torch_perf_flags()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if not data_dir.exists():
        _die(f"data_dir not found: {data_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[*] scan manifest 생성")
    scan_path, speaker_train_quota, total_files = scan_dataset_manifest(data_dir, out_dir)

    # scan 결과에서 바로 train으로 사용
    train_paths = []
    with open(scan_path, "r", encoding="utf-8") as rf:
        for line in rf:
            row = json.loads(line)
            if row["train_eligible"]:
                train_paths.append(Path(row["path"]))

    if len(train_paths) == 0:
        _die("30초 이하 wav가 없습니다.")

    print(f"[*] total wav found        : {total_files}")

    device = pick_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor, model = load_whisper(args.model_id, device, dtype)
    xvec_extractor = XVectorExtractor(device)

    prefix, eos, tok = get_decoder_preset(model, processor, device, args.language)

    shard_writer = PrepareShardWriter(
        out_dir=out_dir,
        shard_token_limit=int(args.shard_token_limit),
        utt_xvec_shard_size=int(args.utt_xvec_shard_size),
    )


    train_utt_id = 0
    keys_dim = None
    xvec_dim = None

    def load_job(p):
        return _load_audio_and_text(p)

    workers = max(1, min(int(args.io_workers), os.cpu_count() or 4))

    print("\n[*] shard 기반 prepare 시작")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        it = executor.map(load_job, train_paths)
        for wav, txt in make_tqdm(it, total=len(train_paths), desc="[3/4] feature/logit extract"):
            x = xvec_extractor(wav).detach().cpu().numpy().astype(np.float32, copy=False)
            shard_writer.add_utt_xvec(x)
            if xvec_dim is None:
                xvec_dim = int(x.shape[0])

            feats = processor.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features
            feats = feats.to(device=device, dtype=dtype, non_blocking=True)
            enc = model.model.encoder(feats, return_dict=True)

            t_ids = tok(txt, add_special_tokens=False).input_ids
            tgt_ids = (
                torch.tensor(t_ids, device=device, dtype=torch.long).view(1, -1)
                if len(t_ids) > 0
                else torch.empty((1, 0), device=device, dtype=torch.long)
            )

            if tgt_ids.shape[1] > 0:
                dec_in = torch.cat([prefix, tgt_ids[:, :-1]], dim=1)
            else:
                dec_in = prefix

            out = model(
                encoder_outputs=enc,
                decoder_input_ids=dec_in,
                output_hidden_states=True,
                return_dict=True,
            )

            plen = prefix.shape[1]
            h = out.decoder_hidden_states[-1][:, plen - 1:plen - 1 + len(t_ids), :].squeeze(0)
            logits = out.logits[:, plen - 1:plen - 1 + len(t_ids), :].squeeze(0)

            if len(t_ids) > 0:
                idx = torch.tensor(t_ids, device=device, dtype=torch.long)
                lt = logits.gather(1, idx.view(-1, 1)).squeeze(1).float().detach().cpu().numpy().astype(np.float32, copy=False)
                lse = torch.logsumexp(logits.float(), dim=-1).detach().cpu().numpy().astype(np.float32, copy=False)
                keys = h.detach().to(torch.float16).cpu().numpy()
                vals = np.asarray(t_ids, dtype=np.int32)
                utt_ids = np.full((len(t_ids),), train_utt_id, dtype=np.int32)

                if keys_dim is None:
                    keys_dim = int(keys.shape[1])

                shard_writer.add_tokens(
                    keys=keys,
                    vals=vals,
                    utt_ids=utt_ids,
                    logit_t=lt,
                    logsumexp=lse,
                )

            train_utt_id += 1

    shard_writer.finalize()
    manifest = shard_writer.build_manifest(
        model_id=args.model_id,
        language=args.language,
        topk=args.topk,
        keys_dim=keys_dim if keys_dim is not None else int(model.config.d_model),
        xvec_dim=xvec_dim if xvec_dim is not None else 40,
        note="Shard-based prepare with in-place JSON transcription normalization and utt_id->xvec lookup",
        index_type="none",
        search_config={
            "ivf_nlist": int(args.ivf_nlist) if args.ivf_nlist > 0 else None,
            "ivf_nprobe": int(args.ivf_nprobe),
            "pq_m": int(args.pq_m) if args.pq_m > 0 else None,
            "pq_nbits": int(args.pq_nbits),
            "faiss_train_samples": int(args.faiss_train_samples),
            "use_faiss_gpu": bool(not args.faiss_cpu_only),
        },
    )

    print("\n[*] FAISS index 준비")
    ds = ShardedDatastore(
        out_dir,
        prefer_faiss=not args.no_faiss,
        cache_size=512,
        ivf_nlist=(int(args.ivf_nlist) if int(args.ivf_nlist) > 0 else None),
        ivf_nprobe=int(args.ivf_nprobe),
        pq_m=(int(args.pq_m) if int(args.pq_m) > 0 else None),
        pq_nbits=int(args.pq_nbits),
        faiss_train_samples=int(args.faiss_train_samples),
        use_faiss_gpu=(False if args.faiss_cpu_only else None),
    )
    ds.build_or_load_index()

    manifest = load_json(out_dir / "manifest.json")
    manifest["index_type"] = ds.index_type
    save_json(out_dir / "manifest.json", manifest)

    print(f"[OK] prepared: {out_dir}")
    print(f" - n_keys: {manifest['n_keys']}")
    print(f" - n_utts: {manifest['n_utts']}")
    print(f" - index_type: {manifest['index_type']}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model_id", default="openai/whisper-large-v3-turbo")
    p.add_argument("--language", default="auto")
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--no_faiss", action="store_true")
    p.add_argument("--io_workers", type=int, default=4)
    p.add_argument("--shard_token_limit", type=int, default=200000)
    p.add_argument("--utt_xvec_shard_size", type=int, default=10000)
    p.add_argument("--ivf_nlist", type=int, default=0)
    p.add_argument("--ivf_nprobe", type=int, default=32)
    p.add_argument("--pq_m", type=int, default=0)
    p.add_argument("--pq_nbits", type=int, default=8)
    p.add_argument("--faiss_train_samples", type=int, default=200000)
    p.add_argument("--faiss_cpu_only", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    cmd_prepare(args)


if __name__ == "__main__":
    main()