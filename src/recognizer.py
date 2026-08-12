"""原生 FunASR 识别器封装。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .text_quality import is_obviously_invalid_asr_text

if TYPE_CHECKING:
    from ..config import FunASRProviderConfig
    from ..protocol import AdapterConfigLike


class StreamingRecognizer:
    """封装原生 FunASR 后端的最小识别接口。"""

    def __init__(
        self,
        adapter_config: "AdapterConfigLike",
        provider_config: "FunASRProviderConfig",
    ) -> None:
        """初始化 FunASR Nano 识别器。"""

        self._adapter_config = adapter_config
        self._provider_config = provider_config
        self._audio_buffer: list[Any] = []
        self._pending_samples = 0
        self._total_samples = 0
        self._last_voice_sample = 0
        self._has_voice = False
        self._text = ""
        self._endpoint = False
        self._finalized = False
        self._recognizer = self._create_recognizer(provider_config)

    @classmethod
    def validate_model_files(cls, config: "FunASRProviderConfig") -> None:
        """校验本地 FunASR 模型目录是否存在。"""

        model = config.asr.model.strip()
        if cls._looks_like_local_path(model) and not Path(model).exists():
            raise FileNotFoundError(f"FunASR 模型路径不存在: asr.model={model}")

    def accept_waveform(self, samples: Any) -> None:
        """缓存一段音频并更新端点状态。"""

        import numpy as np

        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        if array.size == 0:
            return

        self._audio_buffer.append(array)
        self._pending_samples += int(array.size)
        self._total_samples += int(array.size)
        self._trim_audio_buffer()

        rms = float(np.sqrt(np.mean(np.square(array))))
        if rms >= self._provider_config.asr.endpoint_rms_threshold:
            self._last_voice_sample = self._total_samples
            self._has_voice = True
            self._endpoint = False
            self._finalized = False
            return

        silence_samples = self._total_samples - self._last_voice_sample
        silence_ms = silence_samples * 1000 / self._adapter_config.audio.sample_rate
        self._endpoint = bool(self._has_voice and silence_ms >= self._provider_config.asr.endpoint_silence_ms)

    def decode(self) -> None:
        """在累计到推理窗口后执行一次 FunASR 推理。"""

        if not self._provider_config.asr.enable_streaming:
            return

        window_samples = max(
            1,
            int(
                self._adapter_config.audio.sample_rate
                * self._provider_config.asr.inference_window_ms
                / 1000
            ),
        )
        if self._pending_samples < window_samples:
            return
        self._run_inference(is_final=False)
        self._pending_samples = 0

    def get_text(self) -> str:
        """获取当前识别文本。"""

        if self._endpoint and not self._finalized:
            self._run_inference(is_final=True)
            self._finalized = True
        return self._text.strip()

    def get_token_confidences(self) -> list[float]:
        """FunASR AutoModel 未提供此接口，返回空列表。"""

        return []

    def get_avg_confidence(self) -> float | None:
        """FunASR AutoModel 未提供平均 token 置信度。"""

        return None

    def is_endpoint(self) -> bool:
        """返回当前流是否到达端点。"""

        return bool(self._provider_config.asr.enable_endpoint and self._endpoint)

    def has_audio(self) -> bool:
        """返回当前识别段是否已缓存音频。"""

        return bool(self._audio_buffer)

    def mark_endpoint(self) -> None:
        """外部激活结束时强制把当前段标记为端点。"""

        if self._audio_buffer:
            self._endpoint = True

    def reset(self) -> None:
        """重置当前识别状态。"""

        self._audio_buffer = []
        self._pending_samples = 0
        self._total_samples = 0
        self._last_voice_sample = 0
        self._has_voice = False
        self._text = ""
        self._endpoint = False
        self._finalized = False

    def new_stream(self) -> None:
        """开始新的识别段。"""

        self.reset()

    @staticmethod
    def _looks_like_local_path(value: str) -> bool:
        """判断模型配置是否是本地路径而非 ModelScope/Hugging Face 模型 ID。"""

        return bool(
            value.startswith((".", "/", "\\"))
            or ":\\" in value
            or value.startswith("~")
            or Path(value).is_absolute()
        )

    @staticmethod
    def _normalize_device(device: str) -> str:
        """转换配置设备名为 FunASR 可接受的 device 字符串。"""

        value = device.strip().lower()
        if value in {"", "auto"}:
            try:
                import torch

                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        if value in {"gpu", "cuda"}:
            return "cuda:0"
        return value

    @classmethod
    def _create_recognizer(cls, config: "FunASRProviderConfig") -> Any:
        """创建原生 FunASR AutoModel。"""

        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError("缺少依赖 funasr，请先安装插件依赖") from exc
        try:
            import transformers  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "缺少依赖 transformers。Fun-ASR-Nano 使用 HuggingFace tokenizer，"
                "请安装插件依赖或执行: uv pip install transformers tokenizers"
            ) from exc

        asr = config.asr
        remote_code = cls._resolve_remote_code(asr.model, asr.remote_code) if asr.trust_remote_code else asr.remote_code
        if asr.trust_remote_code and remote_code:
            cls._import_remote_code(remote_code)
        kwargs: dict[str, Any] = {
            "model": asr.model,
            "device": cls._normalize_device(asr.device),
            "hub": asr.hub,
            "disable_update": True,
            "check_latest": False,
            "trust_remote_code": asr.trust_remote_code,
            "remote_code": remote_code,
            "disable_pbar": True,
            "log_level": "WARNING",
        }
        if asr.vad_model.strip():
            kwargs["vad_model"] = asr.vad_model.strip()
            kwargs["vad_kwargs"] = {"max_single_segment_time": asr.max_single_segment_time}
        try:
            return AutoModel(**kwargs)
        except AssertionError as exc:
            message = str(exc)
            if "FunASRNano is not registered" in message:
                raise RuntimeError(
                    "当前安装的 funasr 未注册 Fun-ASR-Nano 模型类 FunASRNano。"
                    "Fun-ASR-Nano 官方用法需要 hub='hf'、trust_remote_code=True、"
                    "remote_code='./model.py'，并确保当前网络可访问 Hugging Face；"
                    "如果仍失败，请临时把 asr.model 改为 FunASR 已支持的模型"
                    "（例如 iic/SenseVoiceSmall 或 paraformer-zh）。"
                ) from exc
            raise

    @staticmethod
    def _resolve_remote_code(model: str, remote_code: str) -> str:
        """解析 FunASR 远程模型代码路径。"""

        value = remote_code.strip()
        if not value or value.startswith(("http://", "https://")):
            return value

        path = Path(value)
        if path.is_file():
            return path.resolve().as_posix()

        repo_candidate = Path(__file__).resolve().parents[3] / value
        if repo_candidate.is_file():
            return repo_candidate.resolve().as_posix()

        model_path = Path(model)
        if model_path.exists():
            candidate = model_path / value
            if candidate.is_file():
                return candidate.resolve().as_posix()

        if value in {"model", "model.py", "./model.py", ".\\model.py"}:
            local_vendor_code = (
                Path(__file__).resolve().parents[1]
                / "vendor"
                / "fun_asr"
                / "model.py"
            )
            if local_vendor_code.is_file():
                return local_vendor_code.resolve().as_posix()

            raise RuntimeError(
                "Fun-ASR-Nano 需要官方 FunAudioLLM/Fun-ASR 的 model.py，"
                "但 provider 内置文件不存在: plugins/funasr_asr_provider/vendor/fun_asr/model.py。"
            )

        raise RuntimeError(f"FunASR remote_code 文件不存在: {remote_code}")

    @staticmethod
    def _import_remote_code(remote_code: str) -> None:
        """提前导入 FunASR remote_code，兼容 hub='hf' 不自动导入的版本。"""

        try:
            from funasr.utils.dynamic_import import import_module_from_path
        except ImportError as exc:
            if not remote_code.startswith(("http://", "https://")):
                cls = StreamingRecognizer
                cls._import_local_remote_code(remote_code)
                return
            raise RuntimeError("当前 funasr 版本缺少 remote_code 动态导入能力") from exc
        import_module_from_path(remote_code)

    @staticmethod
    def _import_local_remote_code(remote_code: str) -> None:
        """使用标准库导入本地 remote_code 文件。"""

        import importlib
        import sys

        path = Path(remote_code)
        if not path.is_file():
            raise RuntimeError(f"FunASR remote_code 文件不存在: {remote_code}")
        parent = str(path.parent)
        if parent not in sys.path:
            sys.path.append(parent)
        importlib.import_module(path.stem)

    def _trim_audio_buffer(self) -> None:
        """限制识别窗口长度，避免长时间对话持续增长内存。"""

        import numpy as np

        max_samples = max(
            1,
            int(
                self._adapter_config.audio.sample_rate
                * self._provider_config.asr.max_audio_window_ms
                / 1000
            ),
        )
        total = sum(int(chunk.size) for chunk in self._audio_buffer)
        while self._audio_buffer and total > max_samples:
            first = self._audio_buffer[0]
            overflow = total - max_samples
            if overflow >= first.size:
                self._audio_buffer.pop(0)
                total -= int(first.size)
                continue
            self._audio_buffer[0] = np.ascontiguousarray(first[int(overflow) :])
            break

    def _run_inference(self, *, is_final: bool) -> None:
        """对当前音频窗口调用 FunASR，并保存文本结果。"""

        if not self._audio_buffer:
            return

        import numpy as np
        import torch

        samples = np.ascontiguousarray(np.concatenate(self._audio_buffer), dtype=np.float32)
        if samples.size == 0:
            return

        generate_kwargs = self._build_generate_kwargs(samples, torch, is_final=is_final)
        result = self._recognizer.generate(**generate_kwargs)
        text = self._extract_text(result)
        if text:
            self._text = text

    def _build_generate_kwargs(self, samples: Any, torch_module: Any, *, is_final: bool) -> dict[str, Any]:
        """根据模型类型构造 generate 参数。"""

        asr = self._provider_config.asr
        kwargs: dict[str, Any] = {
            "input": torch_module.from_numpy(samples),
            "cache": {},
            "batch_size_s": asr.batch_size_s,
            "disable_pbar": True,
        }
        if asr.model_type == "funasr_nano":
            kwargs.update(
                {
                    "max_length": asr.max_length,
                    "llm_dtype": asr.llm_dtype,
                    "ctc_only": asr.ctc_only,
                    "is_final": is_final,
                }
            )
        return kwargs

    def _extract_text(self, result: Any) -> str:
        """从 FunASR generate 返回值中提取文本。"""

        if isinstance(result, str):
            return self._postprocess_text(result)
        if isinstance(result, tuple):
            return self._extract_text(result[0] if result else "")
        if isinstance(result, dict):
            return self._postprocess_text(str(result.get("text") or ""))
        if isinstance(result, list):
            parts = []
            for item in result:
                if isinstance(item, dict):
                    text = str(item.get("text") or "").strip()
                    ctc_text = str(item.get("ctc_text") or "").strip()
                    if ctc_text and is_obviously_invalid_asr_text(text).accepted:
                        text = ctc_text
                    elif not text:
                        text = ctc_text
                    if text:
                        parts.append(self._postprocess_text(text))
                elif isinstance(item, str) and item.strip():
                    parts.append(self._postprocess_text(item))
            return "".join(parts).strip()
        return ""

    def _postprocess_text(self, text: str) -> str:
        """按模型类型清洗识别结果文本。"""

        normalized = text.strip()
        if not normalized:
            return ""
        if self._provider_config.asr.model_type != "sensevoice_small":
            return normalized
        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except ImportError:
            return normalized
        return str(rich_transcription_postprocess(normalized)).strip()