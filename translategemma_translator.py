"""TranslateGemma runtime integration.

이 파일이 하는 일:
- Hugging Face에서 TranslateGemma 모델 파일을 프로젝트 `models/` 폴더로 받는다.
- 받은 모델 파일이 최소한 로드 가능한 형태인지 검사한다.
- `transformers`/`torch`/`bitsandbytes`로 모델과 processor를 로드한다.
- 긴 입력을 TranslateGemma 입력 토큰 예산에 맞춰 문단, 문장 순서로 나눈다.
- 나뉜 chunk를 직접 번역하고 번역 캐시를 만든다.

이 파일에 넣지 않는 일:
- Modrinth CSV 읽기/쓰기
- HTML/Markdown 정리
- 불용어 제거와 검색용 토큰화
- tags/categories 메타 토큰 삽입

라이브러리 사용 기준:
- TranslateGemma 모델 카드의 `AutoProcessor.apply_chat_template()`와 `generate()` 예시
- Transformers generation 문서의 greedy decoding 설정
- Transformers bitsandbytes 문서의 `BitsAndBytesConfig`
- huggingface_hub 문서의 `snapshot_download(local_dir=...)`
"""

from __future__ import annotations

import hashlib
import json
import locale
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


LANG_ALIASES = {
    "en": "en",
    "ko": "ko",
    "kr": "ko",
    "ja": "ja",
    "jp": "ja",
    "zh": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh",
}


class TranslationOutputLimitError(RuntimeError):
    """번역 결과가 출력 토큰 한도에 닿아 잘렸을 가능성이 있을 때 사용한다."""


@dataclass(slots=True)
class TranslateGemmaConfig:
    """TranslateGemma 실행 설정.

    preprocessor.py 상단 설정값을 이 객체에 넣어 번역기에 전달한다.
    """

    model_id: str = "google/translategemma-4b-it"
    model_dir: str | None = None
    cache_path: str = "./datasets/translation_cache.json"
    use_translation: bool = True

    # 모델 로드 방식. quantization은 메모리 사용량/속도에 영향을 주고, model_dtype은 비양자화 계산 dtype에 쓰인다.
    quantization: str = "8bit"
    model_dtype: str = "float16"
    device_map: str | int | None = None

    # bitsandbytes 양자화 세부값. 4bit를 쓸 때와 8bit threshold를 조절할 때만 의미가 있다.
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_8bit_threshold: float = 6.0

    # 토큰 예산. 입력+출력 합이 max_total_tokens를 넘지 않도록 chunk 크기와 max_new_tokens 계산에 사용한다.
    max_input_tokens: int = 512
    hard_max_input_tokens: int = 1800
    max_total_tokens: int = 2048
    max_output_tokens: int = 512
    min_output_tokens: int = 64
    output_token_ratio: float = 1.3
    # 실패 원인 검수용 로그. pad-only 출력, 출력 한도 도달, 프롬프트 원문 등을 파일로 남긴다.
    debug_on_failure: bool = True
    debug_to_console: bool = True
    debug_dir: str = "./logs/translation_debug"
    debug_max_chars: int = 4000
    audit_log: bool = True
    audit_dir: str = "./logs/translation_audit"
    log_progress: bool = True
    log_preview_chars: int = 120


def normalize_spacing(text: str) -> str:
    """번역 chunk를 다시 합칠 때 과도한 공백만 정리한다."""
    lines = []
    for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        # 줄 내부의 탭/연속 공백만 줄이고, 줄 자체는 문단 경계로 남긴다.
        line = re.sub(r"[ \t\f\v]+", " ", line).strip()
        lines.append(line)
    text = "\n".join(lines)
    # 빈 줄이 너무 많이 누적되면 결과가 지저분해지므로 두 줄까지만 허용한다.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_lang(lang: str) -> str:
    """언어 코드를 TranslateGemma chat template이 받는 형태로 정규화한다."""
    # Lingua/사용자 입력에서 ko-KR, zh_cn처럼 올 수 있어 소문자와 하이픈 형태로 맞춘다.
    value = str(lang or "").strip().lower().replace("_", "-")
    if not value or value == "unknown":
        return "unknown"
    # zh-cn/zh-tw처럼 세부 지역이 붙은 코드는 모델에 넘기기 전에 대표 언어 코드로 접는다.
    return LANG_ALIASES.get(value, value.split("-", 1)[0])


def safe_model_dir_name(model_id: str) -> str:
    """Hugging Face model id를 Windows 폴더명으로 안전하게 바꾼다."""
    # google/translategemma-12b-it 같은 id의 slash를 폴더명에 안전한 문자열로 바꾼다.
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_id.strip())


def normalize_quantization(value: str | None) -> str:
    """사용자가 넣은 양자화 별칭을 none/8bit/4bit 중 하나로 맞춘다."""
    # 설정 파일에서 int8, 8-bit, nf4처럼 적어도 내부에서는 세 값만 다루게 한다.
    aliases = {
        "": "none",
        "off": "none",
        "false": "none",
        "none": "none",
        "bf16": "none",
        "fp16": "none",
        "8": "8bit",
        "int8": "8bit",
        "8-bit": "8bit",
        "8bit": "8bit",
        "4": "4bit",
        "int4": "4bit",
        "4-bit": "4bit",
        "4bit": "4bit",
        "nf4": "4bit",
    }
    quant = aliases.get(str(value or "none").strip().lower())
    if quant not in {"none", "8bit", "4bit"}:
        # 잘못된 값은 조용히 none으로 넘기면 메모리 사용량이 크게 달라질 수 있어 즉시 실패시킨다.
        raise ValueError("TRANSLATE_QUANTIZATION must be one of: none, 8bit, 4bit")
    return quant


def torch_dtype(torch_module: Any, name: str):
    """문자열 dtype 설정을 torch dtype 객체로 변환한다."""
    # transformers.from_pretrained(dtype=...)에는 문자열이 아니라 torch.float16 같은 객체가 필요하다.
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
        "auto": "auto",
    }
    key = str(name or "float16").lower()
    if key not in mapping:
        # dtype 오타는 모델 로드 방식 자체를 바꾸므로 조기에 명확히 알린다.
        raise ValueError(f"Unknown torch dtype: {name}")
    return mapping[key]


def load_translation_cache(cache_path: str) -> dict[str, str]:
    """번역 캐시 JSON을 읽는다. 없으면 빈 dict를 반환한다."""
    path = Path(cache_path)
    if not path.exists():
        # 첫 실행에는 캐시 파일이 없으므로 빈 캐시로 시작한다.
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_translation_cache(cache: dict[str, str], cache_path: str) -> None:
    """번역 캐시를 디스크에 저장한다."""
    path = Path(cache_path)
    # datasets 폴더가 없을 수도 있으니 저장 직전에 만든다.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def configure_windows_subprocess_text_encoding() -> None:
    """Windows 하위 프로세스 출력 디코딩을 현재 시스템 코드페이지에 맞춘다.

    PyCharm/venv가 Python UTF-8 mode로 실행되면, `subprocess.Popen(text=True)`의
    기본 디코딩도 UTF-8이 된다. 그런데 Windows의 `cmd`, GPU/컴파일러 탐색 도구,
    일부 패키지 진단 명령은 여전히 시스템 코드페이지(cp949 등)로 출력할 수 있다.
    이 불일치가 `subprocess._readerthread`의 UnicodeDecodeError를 만든다.

    모델 계산 결과를 바꾸는 처리는 아니다. Torch/Transformers import 중 실행되는
    환경 탐색용 하위 프로세스가 Windows 출력 인코딩을 제대로 읽게 하는 보정이다.
    """
    if os.name != "nt":
        # 이 문제는 Windows subprocess 디코딩에서만 발생한다.
        return
    if getattr(subprocess.Popen, "_modrinth_encoding_patch", False):
        # 같은 프로세스에서 두 번 패치하지 않도록 표시값을 확인한다.
        return

    getencoding = getattr(locale, "getencoding", None)
    preferred_encoding = (getencoding() if getencoding else None) or locale.getpreferredencoding(False) or "mbcs"
    os.environ["PYTHONUTF8"] = "0"
    os.environ["PYTHONIOENCODING"] = f"{preferred_encoding}:replace"

    original_init = subprocess.Popen.__init__

    def patched_init(self, *args, **kwargs):
        text_mode = kwargs.get("text") or kwargs.get("universal_newlines")
        if text_mode:
            # text=True인데 encoding이 없으면 현재 Windows 코드페이지를 명시해 디코딩 오류를 줄인다.
            current_encoding = kwargs.get("encoding")
            if current_encoding is None:
                kwargs["encoding"] = preferred_encoding
            if kwargs.get("errors") is None:
                kwargs["errors"] = "replace"
        return original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = patched_init
    subprocess.Popen._modrinth_encoding_patch = True


class TranslateGemmaTranslator:
    """TranslateGemma 모델 로드와 번역 실행을 맡는 객체."""

    def __init__(self, config: TranslateGemmaConfig):
        self.config = config
        # 모델과 processor는 첫 번역 때 load()에서 채운 뒤 재사용한다.
        self._model = None
        self._processor = None

    def model_dir(self) -> Path:
        """현재 모델을 저장하거나 로드할 프로젝트 내부 폴더를 반환한다."""
        if self.config.model_dir:
            # 사용자가 직접 모델 폴더를 지정하면 그 경로를 우선한다.
            return Path(self.config.model_dir)
        # 기본값은 프로젝트 models/ 아래에 model_id 기반 폴더를 만든다.
        return Path("./models") / safe_model_dir_name(self.config.model_id)

    def validate_local_model_files(self, model_dir: Path) -> tuple[bool, str]:
        """중단된 다운로드를 조기에 잡기 위한 최소 파일 검증.

        Hugging Face Hub는 자체 캐시에서는 etag/commit 기반으로 파일을 관리하지만,
        `local_dir`로 받은 폴더를 직접 쓰는 경우 로드 전에 index가 가리키는 shard가
        실제로 있는지 확인해 주는 편이 오류 메시지가 훨씬 명확하다.
        """
        config_json = model_dir / "config.json"
        if not config_json.exists():
            # config가 없으면 transformers가 모델 종류를 판단할 수 없다.
            return False, "config.json not found"

        index_json = model_dir / "model.safetensors.index.json"
        if index_json.exists():
            try:
                # shard 모델은 index 파일의 weight_map이 실제 safetensors 파일명을 알려준다.
                index = json.loads(index_json.read_text(encoding="utf-8"))
                required = sorted(set(index.get("weight_map", {}).values()))
            except json.JSONDecodeError as exc:
                return False, f"model.safetensors.index.json is invalid: {exc}"

            # 다운로드가 중간에 끊기면 파일이 없거나 0바이트로 남을 수 있다.
            missing = [name for name in required if not (model_dir / name).exists()]
            empty = [name for name in required if (model_dir / name).exists() and (model_dir / name).stat().st_size == 0]
            if missing or empty:
                return False, f"missing={missing[:3]}, empty={empty[:3]}"
            return True, "ok"

        safetensors = list(model_dir.glob("*.safetensors"))
        if not safetensors:
            # index가 없는 단일 파일 모델이라도 safetensors 가중치는 있어야 한다.
            return False, "no safetensors weights found"
        empty = [path.name for path in safetensors if path.stat().st_size == 0]
        if empty:
            return False, f"empty safetensors files: {empty[:3]}"
        return True, "ok"

    def ensure_model_dir(self) -> str:
        """모델 폴더를 준비하고, 불완전하면 다시 다운로드를 시도한다."""
        configure_windows_subprocess_text_encoding()

        model_dir = self.model_dir()
        is_valid, reason = self.validate_local_model_files(model_dir) if model_dir.exists() else (False, "not downloaded")
        if is_valid:
            # 이미 정상 다운로드된 모델이면 네트워크를 쓰지 않고 바로 로컬 경로를 반환한다.
            return str(model_dir)

        from huggingface_hub import snapshot_download

        print(f"[translator] model download/repair: {self.config.model_id} ({reason})")
        model_dir.mkdir(parents=True, exist_ok=True)
        # local_dir에 직접 받아 이후 from_pretrained(local_files_only=True)로 로드한다.
        snapshot_download(repo_id=self.config.model_id, local_dir=str(model_dir))

        is_valid, reason = self.validate_local_model_files(model_dir)
        if not is_valid:
            # 재다운로드 후에도 파일이 불완전하면 모델 로드 전에 명확히 실패시킨다.
            raise RuntimeError(f"TranslateGemma model download is incomplete: {model_dir} ({reason})")
        return str(model_dir)

    def build_quantization_config(self, torch_module: Any):
        """Transformers `from_pretrained`에 넘길 BitsAndBytesConfig를 만든다."""
        quant = normalize_quantization(self.config.quantization)
        if quant == "none":
            # 양자화를 쓰지 않으면 transformers에 quantization_config를 넘기지 않는다.
            return None

        from transformers import BitsAndBytesConfig

        if quant == "8bit":
            # 8bit는 load_in_8bit와 threshold만 지정한다.
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=self.config.bnb_8bit_threshold,
            )

        # 4bit는 nf4/fp4 종류와 계산 dtype, double quant 여부까지 지정한다.
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch_dtype(torch_module, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )

    def effective_device_map(self, torch_module: Any):
        """기본 장치 배치.

        CUDA가 있으면 한 GPU에 전부 올린다. 이 설정은 `auto` offload 때문에
        4bit 모델이 CPU/디스크로 밀리며 실패하는 상황을 피하기 위한 기본값이다.
        """
        if self.config.device_map is not None:
            # 사용자가 직접 device_map을 넣으면 자동 판단보다 우선한다.
            return self.config.device_map
        return 0 if torch_module.cuda.is_available() else "cpu"

    def load(self):
        """TranslateGemma processor와 모델을 한 번만 로드한다."""
        if not self.config.use_translation:
            # 번역을 끈 상태에서는 호출부가 None을 받을 수 있게 한다.
            return None, None
        if self._model is not None:
            # 이미 로드된 모델은 다시 from_pretrained하지 않는다.
            return self._model, self._processor

        configure_windows_subprocess_text_encoding()

        import torch
        from transformers import AutoProcessor

        model_dir = self.ensure_model_dir()
        model_kwargs = {
            "device_map": self.effective_device_map(torch),
            "dtype": torch_dtype(torch, self.config.model_dtype),
            "local_files_only": True,
        }
        quantization_config = self.build_quantization_config(torch)
        if quantization_config is not None:
            # 양자화 설정이 있을 때만 from_pretrained 인자에 추가한다.
            model_kwargs["quantization_config"] = quantization_config

        # processor는 chat template과 tokenizer를 함께 제공한다.
        self._processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)

        from transformers import AutoModelForImageTextToText

        # 공식 사용법에 맞춰 image-text-to-text 모델 클래스로 로드한다.
        self._model = AutoModelForImageTextToText.from_pretrained(model_dir, **model_kwargs)

        print("[translator]")
        print(f"- model: {self.config.model_id}")
        print("- model_loader: AutoModelForImageTextToText")
        print(f"- quantization: {normalize_quantization(self.config.quantization)}")
        print(f"- generation_pad_token_id: {self._model.generation_config.pad_token_id}")
        print(f"- generation_eos_token_id: {self._model.generation_config.eos_token_id}")
        if hasattr(self._model, "get_memory_footprint"):
            print(f"- memory: {self._model.get_memory_footprint() / (1024 ** 3):.2f} GiB")
        if torch.cuda.is_available():
            print(f"- cuda_allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GiB")

        return self._model, self._processor

    def translation_messages(self, text: str, source_lang: str, target_lang: str) -> list[dict[str, Any]]:
        """TranslateGemma chat template 입력을 만든다."""
        # TranslateGemma processor는 source/target 언어 코드와 원문을 content 안에서 읽는다.
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": normalize_lang(source_lang),
                        "target_lang_code": normalize_lang(target_lang),
                        "text": text,
                    }
                ],
            }
        ]

    def count_input_tokens(self, text: str, source_lang: str, target_lang: str) -> int:
        """chat template 적용 후 실제 입력 토큰 수를 계산한다."""
        _model, processor = self.load()
        # 원문 글자 수가 아니라 chat template과 generation prompt까지 포함한 모델 입력 토큰을 센다.
        inputs = processor.apply_chat_template(
            self.translation_messages(text, source_lang, target_lang),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return int(inputs["input_ids"].shape[-1])

    def split_paragraphs(self, text: str) -> list[str]:
        """빈 줄 기준 문단 분리. 문맥 유지를 위해 최우선 분할 단위로 쓴다."""
        # 전처리기에서 description/body 경계도 빈 줄로 넣으므로 이 경계를 우선 보존한다.
        return [p.strip() for p in re.split(r"\n\s*\n+", str(text)) if p.strip()]

    def split_sentences(self, text: str) -> list[str]:
        """단일 문단이 토큰 예산을 넘을 때만 문장 단위로 나눈다."""
        # 영어와 CJK 문장부호 뒤 공백을 문장 경계로 본다.
        pieces = re.split(r"(?<=[.!?。！？])\s+", str(text).strip())
        return [piece.strip() for piece in pieces if piece.strip()] or [str(text).strip()]

    def split_soft_boundaries(self, text: str) -> list[str]:
        """문장 부호가 부족한 모드 목록/파일 목록을 약한 경계로 나눈다."""
        # 쉼표/세미콜론/대괄호/하이픈 목록 같은 약한 구분자는 문장이 없을 때 보조 분할 기준이 된다.
        pieces = re.split(
            r"(?<=[,;:，；：、])\s+|\s+-\s+|(?<=\])\s+|\s+(?=\[)",
            str(text).strip(),
        )
        return [piece.strip() for piece in pieces if piece.strip()]

    def split_in_half(self, text: str) -> list[str]:
        """뚜렷한 구분자가 없는 긴 덩어리를 마지막 수단으로 반으로 나눈다."""
        words = str(text).split()
        if len(words) > 1:
            # 공백이 있는 텍스트는 단어 개수 기준으로 반씩 나눠 단어 중간 절단을 피한다.
            midpoint = len(words) // 2
            return [" ".join(words[:midpoint]), " ".join(words[midpoint:])]

        text = str(text).strip()
        if len(text) <= 1:
            return [text] if text else []
        # 공백도 없는 긴 문자열은 더 좋은 경계가 없으므로 문자 수 기준으로 나눈다.
        midpoint = len(text) // 2
        return [text[:midpoint].strip(), text[midpoint:].strip()]

    def split_oversized_segment(self, segment: str) -> list[str]:
        """입력 예산을 넘는 단일 segment를 문장, 약한 경계, 반분 순서로 쪼갠다."""
        # 앞쪽 splitter일수록 의미 경계를 덜 깨므로 성공하면 즉시 그 결과를 사용한다.
        for splitter in (self.split_sentences, self.split_soft_boundaries, self.split_in_half):
            pieces = splitter(segment)
            if len(pieces) > 1:
                return pieces
        return [segment]

    def chunk_input_token_limit(self) -> int:
        """번역 출력까지 고려해 모델 카드의 2K 예산 안에 들어갈 입력 chunk 크기를 잡는다."""
        # output_token_ratio만큼 출력이 늘어난다고 가정해, 입력이 너무 커지지 않도록 역산한다.
        ratio_limit = int((self.config.max_total_tokens - self.config.min_output_tokens) / (1 + self.config.output_token_ratio))
        return max(1, min(self.config.max_input_tokens, ratio_limit))

    def pack_segments(self, segments: list[str], separator: str, source_lang: str, target_lang: str) -> list[str]:
        """문단/문장을 입력 토큰 예산 이하의 chunk 목록으로 묶는다."""
        chunks: list[str] = []
        current = ""
        token_limit = self.chunk_input_token_limit()

        for segment in segments:
            # 문단 하나가 이미 예산을 넘으면 current와 합치지 않고 더 작은 문장/구간으로 나눈다.
            if self.count_input_tokens(segment, source_lang, target_lang) > token_limit:
                if current:
                    chunks.append(current)
                    current = ""
                pieces = self.split_oversized_segment(segment)
                if len(pieces) == 1:
                    print(f"[translation skip: cannot split over token budget] {segment[:120]}")
                    continue
                chunks.extend(self.pack_segments(pieces, " ", source_lang, target_lang))
                continue

            candidate = segment if not current else f"{current}{separator}{segment}"
            if self.count_input_tokens(candidate, source_lang, target_lang) <= token_limit:
                # 현재 chunk에 segment를 붙여도 예산 안이면 계속 합친다.
                current = candidate
            else:
                # 붙이면 예산을 넘는 순간 현재 chunk를 확정하고 새 chunk를 시작한다.
                if current:
                    chunks.append(current)
                current = segment

        if current:
            chunks.append(current)
        return chunks

    def split_for_translation(self, text: str, source_lang: str, target_lang: str) -> list[str]:
        """전체 원문을 문단 우선, 문장 fallback 방식으로 chunk화한다."""
        # 외부에서 chunk만 필요할 때 쓰는 얇은 진입점이다.
        return self.pack_segments(self.split_paragraphs(text), "\n\n", source_lang, target_lang)

    def preview_text(self, text: str) -> str:
        """진행 로그에 넣을 짧은 미리보기 문자열을 만든다."""
        # 로그 한 줄에 넣기 위해 줄바꿈을 공백으로 바꾼다.
        text = normalize_spacing(text).replace("\n", " ")
        limit = self.config.log_preview_chars
        if limit and len(text) > limit:
            # 긴 원문은 앞부분만 보여주고 실제 전체 내용은 audit/debug 파일에서 확인한다.
            return text[:limit] + "..."
        return text

    def token_stats(self, token_counts: list[int]) -> str:
        """chunk 토큰 수 요약을 사람이 읽기 좋게 만든다."""
        if not token_counts:
            # 빈 리스트에서도 로그 포맷이 깨지지 않게 0 요약을 반환한다.
            return "min=0 avg=0 max=0"
        avg = sum(token_counts) / len(token_counts)
        return f"min={min(token_counts)} avg={avg:.1f} max={max(token_counts)}"

    def char_stats(self, chunks: list[str]) -> str:
        """chunk 문자 수 요약을 사람이 읽기 좋게 만든다."""
        if not chunks:
            # chunk가 없을 때도 split 로그가 같은 형식을 유지한다.
            return "min=0 avg=0 max=0"
        lengths = [len(chunk) for chunk in chunks]
        avg = sum(lengths) / len(lengths)
        return f"min={min(lengths)} avg={avg:.1f} max={max(lengths)}"

    def max_new_token_cap(self, input_tokens: int) -> int:
        """입력+출력이 모델 카드의 2K 예산을 넘지 않도록 출력 상한을 계산한다."""
        # 남은 토큰 수가 실제 출력 상한이다. max_output_tokens보다 커도 모델 전체 예산을 넘길 수 없다.
        remaining = self.config.max_total_tokens - input_tokens
        return max(self.config.min_output_tokens, min(self.config.max_output_tokens, remaining))

    def max_new_tokens_for(self, input_tokens: int) -> int:
        """입력 길이에 맞춰 chunk 하나의 기본 출력 토큰 상한을 잡는다."""
        # 기본값은 입력 토큰 * ratio지만, 최종적으로는 max_total_tokens에서 남은 예산으로 한 번 더 자른다.
        budget = int(input_tokens * self.config.output_token_ratio)
        budget = max(self.config.min_output_tokens, budget)
        return min(budget, self.max_new_token_cap(input_tokens))

    def clipped_debug_text(self, text: str) -> str:
        """콘솔에 출력할 디버그 텍스트를 제한한다."""
        text = str(text)
        limit = self.config.debug_max_chars
        if limit and len(text) > limit:
            # 전체 내용은 파일에 저장하고 콘솔에는 앞부분만 보여준다.
            return text[:limit] + f"\n...[truncated debug text: {len(text) - limit} chars omitted]"
        return text

    def write_translation_debug(
        self,
        reason: str,
        source_lang: str,
        target_lang: str,
        chunk: str,
        prompt_text: str,
        generated_raw: str,
        generated_clean: str,
        input_tokens: int,
        output_tokens: int,
        max_new_tokens: int,
        eos_token_ids,
    ) -> None:
        """번역 실패 시 실제 입력/프롬프트/생성 결과를 화면과 파일에 남긴다."""
        if not self.config.debug_on_failure:
            # 실패 디버그가 꺼져 있으면 번역 흐름을 방해하지 않고 바로 돌아간다.
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 같은 시각에 여러 실패가 나도 파일명이 겹치지 않게 실패 이유와 chunk 내용으로 짧은 해시를 만든다.
        digest = hashlib.sha1(f"{reason}\n{source_lang}\n{target_lang}\n{chunk}".encode("utf-8")).hexdigest()[:12]
        debug_path = Path(self.config.debug_dir) / f"{timestamp}_{digest}.txt"
        debug_path.parent.mkdir(parents=True, exist_ok=True)

        # raw는 특수 토큰 포함 출력, clean은 특수 토큰 제거 출력이다. pad-only 문제를 보려면 둘 다 필요하다.
        content = "\n".join([
            "=== TRANSLATION DEBUG ===",
            f"reason: {reason}",
            f"source_lang: {source_lang}",
            f"target_lang: {target_lang}",
            f"input_tokens: {input_tokens}",
            f"output_tokens: {output_tokens}",
            f"max_new_tokens: {max_new_tokens}",
            f"eos_token_ids: {eos_token_ids}",
            "",
            "=== SOURCE CHUNK ===",
            chunk,
            "",
            "=== RENDERED PROMPT ===",
            prompt_text,
            "",
            "=== GENERATED RAW / SPECIAL TOKENS KEPT ===",
            generated_raw,
            "",
            "=== GENERATED CLEAN / SPECIAL TOKENS REMOVED ===",
            generated_clean,
            "",
        ])
        debug_path.write_text(content, encoding="utf-8")

        if self.config.debug_to_console:
            # 콘솔에는 원인과 짧은 원문/출력만 보여주고, 전체는 debug_file에서 확인한다.
            print("\n[translation debug]")
            print(f"- reason: {reason}")
            print(f"- source={source_lang} target={target_lang}")
            print(f"- input_tokens={input_tokens} output_tokens={output_tokens} max_new_tokens={max_new_tokens}")
            print(f"- eos_token_ids={eos_token_ids}")
            print(f"- debug_file: {debug_path}")
            print("[source chunk]")
            print(self.clipped_debug_text(chunk))
            print("[generated clean]")
            print(self.clipped_debug_text(generated_clean))
            print("[generated raw]")
            print(self.clipped_debug_text(generated_raw))

    def write_translation_audit(
        self,
        source_lang: str,
        target_lang: str,
        label: str,
        source_text: str,
        translated_text: str,
        chunks: list[str],
        translated_chunks: list[str],
        total_elapsed: float,
        cache_hit: bool = False,
    ) -> None:
        """번역 입출력을 항상 파일로 남겨 언어 감지와 번역 품질을 수동 검수할 수 있게 한다."""
        if not self.config.audit_log:
            # 감사 로그를 끄면 번역 결과만 반환하고 파일은 만들지 않는다.
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 같은 row가 재시도되어도 원문/언어/label 조합이 파일명에 반영되도록 해시를 붙인다.
        digest = hashlib.sha1(f"{source_lang}\n{target_lang}\n{label}\n{source_text}".encode("utf-8")).hexdigest()[:12]
        audit_path = Path(self.config.audit_dir) / f"{timestamp}_{digest}.txt"
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        # 파일 앞부분에는 전체 원문/전체 번역문을 넣어 사람이 바로 비교할 수 있게 한다.
        content = [
            "=== TRANSLATION AUDIT ===",
            f"label: {label}",
            f"source_lang: {source_lang}",
            f"target_lang: {target_lang}",
            f"cache_hit: {int(cache_hit)}",
            f"chunks: {len(chunks)}",
            f"elapsed_sec: {total_elapsed:.1f}",
            f"source_chars: {len(source_text)}",
            f"translated_chars: {len(translated_text)}",
            "",
            "=== SOURCE TEXT ===",
            source_text,
            "",
            "=== TRANSLATED TEXT ===",
            translated_text,
            "",
        ]
        for index, (chunk, translated_chunk) in enumerate(zip(chunks, translated_chunks), 1):
            # 뒤쪽에는 chunk별 원문/번역을 붙여 어느 chunk에서 품질 문제가 생겼는지 볼 수 있게 한다.
            content.extend([
                f"=== CHUNK {index}/{len(chunks)} SOURCE ===",
                chunk,
                "",
                f"=== CHUNK {index}/{len(chunks)} TRANSLATED ===",
                translated_chunk,
                "",
            ])
        audit_path.write_text("\n".join(content), encoding="utf-8")
        if self.config.log_progress:
            print(f"[translation audit] {audit_path}")

    def translate_chunk_once(self, chunk: str, source_lang: str, target_lang: str, max_new_tokens: int) -> str:
        """chunk 하나를 지정된 출력 토큰 한도로 한 번 번역한다."""
        import torch

        model, processor = self.load()
        inputs = processor.apply_chat_template(
            self.translation_messages(chunk, source_lang, target_lang),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # input_len은 prompt 부분 길이다. generate 결과에서 이 길이 이후만 실제 번역 출력으로 잘라낸다.
        input_len = int(inputs["input_ids"].shape[-1])
        eos_token_ids = getattr(model.generation_config, "eos_token_id", None)
        prompt_text = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)

        # processor 출력은 CPU tensor이므로 모델이 올라간 device로 옮긴다.
        inputs = inputs.to(model.device)

        generate_kwargs = {
            **inputs,
            "do_sample": False,
            # 이 값은 max_new_tokens_for/max_new_token_cap에서 계산된 출력 전용 토큰 한도다.
            "max_new_tokens": max_new_tokens,
        }

        with torch.inference_mode():
            output = model.generate(**generate_kwargs)

        generated = output[0][input_len:]
        generated_raw = processor.decode(generated, skip_special_tokens=False).strip()
        generated_clean = processor.decode(generated, skip_special_tokens=True).strip()
        # raw는 있는데 clean이 비면 <pad> 같은 특수 토큰만 생성한 상황이다. 정상 번역으로 쓰면 안 된다.
        if generated_raw and not generated_clean and generated_raw.replace("<pad>", "").strip() == "":
            self.write_translation_debug(
                reason="pad_only_output",
                source_lang=source_lang,
                target_lang=target_lang,
                chunk=chunk,
                prompt_text=prompt_text,
                generated_raw=generated_raw,
                generated_clean=generated_clean,
                input_tokens=input_len,
                output_tokens=len(generated),
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
            )
            raise RuntimeError("translation generated only <pad> tokens")
        # 출력 길이가 한도에 닿았으면 문장이 잘렸을 가능성이 높아 더 큰 한도/더 작은 chunk로 재시도한다.
        if len(generated) >= max_new_tokens:
            self.write_translation_debug(
                reason="output_token_limit_hit",
                source_lang=source_lang,
                target_lang=target_lang,
                chunk=chunk,
                prompt_text=prompt_text,
                generated_raw=generated_raw,
                generated_clean=generated_clean,
                input_tokens=input_len,
                output_tokens=len(generated),
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
            )
            raise TranslationOutputLimitError(
                f"translation output token limit hit: input_tokens={input_len}, max_new_tokens={max_new_tokens}"
            )
        return generated_clean

    def split_chunk_for_output_retry(self, chunk: str) -> list[str]:
        """출력 한도에 걸린 chunk를 더 작은 단위로 나눈다."""
        # 문단이 여러 개면 문단별 번역이 가장 안전하므로 먼저 시도한다.
        paragraphs = self.split_paragraphs(chunk)
        if len(paragraphs) > 1:
            return paragraphs

        # 문단 하나짜리라면 문장 단위로 나눠 출력 길이를 줄인다.
        sentences = self.split_sentences(chunk)
        if len(sentences) > 1:
            return sentences

        # 문장 경계도 없으면 약한 경계/반분 로직으로 마지막 분할을 시도한다.
        return self.split_oversized_segment(chunk)

    def translate_chunk(self, chunk: str, source_lang: str, target_lang: str, depth: int = 0) -> str:
        """chunk 하나를 번역한다.

        출력 한도에 걸리면 `max_output_tokens`까지 한 번 올려 재시도하고,
        그래도 잘리면 chunk를 더 작게 나눠 재귀적으로 다시 번역한다.
        """
        input_tokens = self.count_input_tokens(chunk, source_lang, target_lang)
        token_limit = self.chunk_input_token_limit()
        if input_tokens > token_limit:
            # pack 이후에도 입력이 너무 크면 문장/구간 단위로 다시 쪼개 번역한다.
            smaller_chunks = self.pack_segments(self.split_oversized_segment(chunk), " ", source_lang, target_lang)
            if len(smaller_chunks) > 1 and depth < 8:
                print(f"[translation retry: split oversized input] input_tokens={input_tokens} chunks={len(smaller_chunks)}")
                translated = [
                    self.translate_chunk(part, source_lang, target_lang, depth=depth + 1)
                    for part in smaller_chunks
                    if part.strip()
                ]
                return normalize_spacing("\n\n".join(part for part in translated if part))

        if input_tokens > self.config.hard_max_input_tokens:
            # hard limit은 더 이상 안전하게 재시도할 수 없는 입력 크기다.
            raise RuntimeError(
                f"translation input hard limit exceeded: input_tokens={input_tokens}, "
                f"hard_max_input_tokens={self.config.hard_max_input_tokens}"
            )

        first_limit = self.max_new_tokens_for(input_tokens)
        retry_limits = [first_limit]
        retry_limit = self.max_new_token_cap(input_tokens)
        if retry_limit not in retry_limits:
            # 첫 출력 한도가 ratio 기준이라 작았다면, 남은 전체 예산까지 한 번 더 시도한다.
            retry_limits.append(retry_limit)

        last_error: Exception | None = None
        for limit in retry_limits:
            try:
                # 같은 chunk를 현재 출력 한도로 한 번 시도한다.
                return self.translate_chunk_once(chunk, source_lang, target_lang, limit)
            except TranslationOutputLimitError as exc:
                # 출력 한도에 닿은 실패만 더 큰 한도로 재시도한다.
                last_error = exc
                if limit < retry_limit:
                    print(f"[translation retry: larger output] {limit}->{retry_limit}")
                    continue

        smaller_chunks = self.split_chunk_for_output_retry(chunk)
        if len(smaller_chunks) > 1 and depth < 3:
            # 출력 한도를 키워도 잘리면 chunk 자체를 더 작게 나눠 각각 번역한다.
            print(f"[translation retry: split smaller] {len(smaller_chunks)} chunks")
            translated = [
                self.translate_chunk(part, source_lang, target_lang, depth=depth + 1)
                for part in smaller_chunks
                if part.strip()
            ]
            return normalize_spacing("\n\n".join(part for part in translated if part))

        # 더 나눌 수 없거나 재귀 제한에 닿으면 마지막 실패 원인을 호출부로 올린다.
        raise last_error or RuntimeError("translation failed")

    def cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """모델/토큰 정책이 바뀌면 달라지는 번역 캐시 키를 만든다."""
        # 원문 전체를 키에 넣지 않고 hash만 넣어 JSON 캐시가 과하게 커지지 않게 한다.
        # model_id/quantization/token 정책이 바뀌면 같은 원문도 다른 결과가 될 수 있어 키에 포함한다.
        raw = {
            "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "source_lang": normalize_lang(source_lang),
            "target_lang": normalize_lang(target_lang),
            "model_id": self.config.model_id,
            "model_loader": "AutoModelForImageTextToText",
            "quantization": normalize_quantization(self.config.quantization),
            "max_input_tokens": self.config.max_input_tokens,
            "hard_max_input_tokens": self.config.hard_max_input_tokens,
            "max_total_tokens": self.config.max_total_tokens,
            "max_output_tokens": self.config.max_output_tokens,
            "chunking": "paragraph-then-sentence-v1",
            "generation_stop": "official-generate-config-v2-context-budget",
        }
        return hashlib.sha256(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()

    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        cache: dict[str, str],
        label: str = "",
        raise_on_failure: bool = False,
    ) -> str:
        """긴 텍스트를 chunk 단위로 직접 번역하고 캐시한다."""
        started_at = time.perf_counter()
        text = str(text).strip()
        source_lang = normalize_lang(source_lang)
        target_lang = normalize_lang(target_lang)
        label_text = f" {label}" if label else ""

        if not text or not self.config.use_translation:
            # 번역이 꺼져 있거나 입력이 비었으면 모델을 로드하지 않는다.
            return text if source_lang == target_lang else ""
        if source_lang == target_lang:
            # 이미 목표 언어면 원문 그대로 반환한다.
            return text
        if source_lang == "unknown":
            # TranslateGemma에는 source_lang_code가 필요하므로 unknown은 호출부에서 처리해야 한다.
            raise ValueError("TranslateGemma requires a known source language code")

        key = self.cache_key(text, source_lang, target_lang)
        if key in cache:
            # 같은 원문/언어/모델 설정이면 번역 모델을 다시 돌리지 않고 캐시 결과를 쓴다.
            if self.config.log_progress:
                print(f"[translation cache hit]{label_text} {source_lang}->{target_lang} chars={len(text)}")
            self.write_translation_audit(
                source_lang=source_lang,
                target_lang=target_lang,
                label=label,
                source_text=text,
                translated_text=cache[key],
                chunks=[text],
                translated_chunks=[cache[key]],
                total_elapsed=time.perf_counter() - started_at,
                cache_hit=True,
            )
            return cache[key]

        paragraphs = self.split_paragraphs(text)
        # 문단별 토큰 수를 먼저 재서, 분할 전 입력이 얼마나 큰지 로그로 확인한다.
        paragraph_tokens = [self.count_input_tokens(paragraph, source_lang, target_lang) for paragraph in paragraphs]
        # 큰 문단 수는 chunk 분할이 제대로 되는지 확인하기 위한 진행 로그 지표다.
        oversized_paragraphs = sum(tokens > self.config.max_input_tokens for tokens in paragraph_tokens)

        if self.config.log_progress:
            print(
                f"[translation start]{label_text} {source_lang}->{target_lang} "
                f"chars={len(text)} paragraphs={len(paragraphs)} "
                f"paragraph_tokens=({self.token_stats(paragraph_tokens)}) "
                f"oversized_paragraphs={oversized_paragraphs} "
                f"preview={self.preview_text(text)}"
            )

        # 문단을 token_limit 이하 chunk로 묶는다. chunk 사이에는 문단 경계인 "\n\n"을 유지한다.
        chunks = self.pack_segments(paragraphs, "\n\n", source_lang, target_lang)
        # chunk별 토큰 수는 진행 로그와 max_new_tokens 계산에 재사용한다.
        chunk_tokens = [self.count_input_tokens(chunk, source_lang, target_lang) for chunk in chunks]

        if self.config.log_progress:
            print(
                f"[translation split]{label_text} {source_lang}->{target_lang} "
                f"chunks={len(chunks)} chunk_tokens=({self.token_stats(chunk_tokens)}) "
                f"chunk_chars=({self.char_stats(chunks)})"
            )

        translated: list[str] = []
        failed_chunks = 0
        for index, chunk in enumerate(chunks, 1):
            # chunk별로 실패를 세어 row 전체를 complete=1로 잘못 기록하지 않게 한다.
            start = time.perf_counter()
            # chunk_tokens를 미리 계산했지만 방어적으로 index가 어긋나면 즉시 다시 계산한다.
            input_tokens = chunk_tokens[index - 1] if index - 1 < len(chunk_tokens) else self.count_input_tokens(chunk, source_lang, target_lang)
            # 로그용 출력 한도다. 실제 translate_chunk 내부에서도 같은 방식으로 다시 계산한다.
            max_new_tokens = self.max_new_tokens_for(input_tokens)
            if self.config.log_progress:
                print(
                    f"[translation chunk start]{label_text} {source_lang}->{target_lang} "
                    f"chunk={index}/{len(chunks)} chars={len(chunk)} "
                    f"input_tokens={input_tokens} max_new_tokens={max_new_tokens} "
                    f"preview={self.preview_text(chunk)}"
                )
            try:
                result = self.translate_chunk(chunk, source_lang, target_lang)
            except Exception as exc:
                # 한 chunk라도 실패하면 최종 translate_text 아래쪽에서 실패로 처리된다.
                failed_chunks += 1
                print(
                    "[translation chunk failed] "
                    f"{index}/{len(chunks)} source={source_lang} target={target_lang} "
                    f"elapsed={time.perf_counter() - start:.1f}s error={exc}"
                )
                continue
            elapsed = time.perf_counter() - start
            # 정확한 토큰 수가 아니라 사람이 보기 위한 대략적인 출력 길이 지표다.
            output_tokens_estimate = len(result.split())
            print(
                f"[translation chunk done]{label_text} {source_lang}->{target_lang} "
                f"chunk={index}/{len(chunks)} elapsed={elapsed:.1f}s "
                f"output_chars={len(result)} output_words~={output_tokens_estimate}"
            )
            if result:
                # 빈 결과는 합치지 않는다. 실패는 예외로 처리되고, 빈 문자열은 의미 있는 번역 결과가 아니다.
                translated.append(result)

        # chunk 번역을 원래 순서대로 빈 줄로 합치고 공백을 정리한다.
        result_text = normalize_spacing("\n\n".join(translated))
        total_elapsed = time.perf_counter() - started_at
        if failed_chunks:
            # 일부 chunk만 성공한 번역은 완전한 결과가 아니므로 캐시에 저장하지 않는다.
            print(
                f"[translation done with failures]{label_text} {source_lang}->{target_lang} "
                f"elapsed={total_elapsed:.1f}s chunks={len(chunks)} failed={failed_chunks} "
                f"output_chars={len(result_text)} cache_saved=False"
            )
            if raise_on_failure:
                # 전처리기는 raise_on_failure=True로 호출해 row를 complete=0으로 기록하게 한다.
                raise RuntimeError(
                    f"translation incomplete: chunks={len(chunks)} failed={failed_chunks} "
                    f"source={source_lang} target={target_lang}"
                )
            return result_text

        # 모든 chunk가 성공한 경우에만 캐시에 저장한다.
        cache[key] = result_text
        # 성공 번역도 수동 검수할 수 있게 audit 파일을 남긴다.
        self.write_translation_audit(
            source_lang=source_lang,
            target_lang=target_lang,
            label=label,
            source_text=text,
            translated_text=result_text,
            chunks=chunks,
            translated_chunks=translated,
            total_elapsed=total_elapsed,
            cache_hit=False,
        )
        if self.config.log_progress:
            # 완료 로그에는 chunk 수와 캐시 저장 여부를 남겨 전처리 진행 상태를 확인한다.
            print(
                f"[translation done]{label_text} {source_lang}->{target_lang} "
                f"elapsed={total_elapsed:.1f}s chunks={len(chunks)} failed=0 "
                f"output_chars={len(result_text)} cache_saved=True"
            )
        return result_text
