"""FunASR Provider 的配置校验与运行时工厂。"""

from __future__ import annotations

from .config import FunASRProviderConfig
from .protocol import AdapterConfigLike

from .src.audio_source import MicrophoneAudioSource
from .src.recognizer import StreamingRecognizer


class FunASRProvider:
    """根据套件配置创建麦克风音频源和 FunASR 识别器。"""

    provider_name = "funasr"

    def __init__(self, config: FunASRProviderConfig) -> None:
        """保存 provider 自己的配置。"""

        self._config = config

    def validate_config(self, config: AdapterConfigLike) -> None:
        """校验 FunASR 模型配置。"""

        _ = config
        StreamingRecognizer.validate_model_files(self._config)

    def create_audio_source(
        self,
        config: AdapterConfigLike,
    ) -> MicrophoneAudioSource:
        """创建默认麦克风音频源。"""

        return MicrophoneAudioSource(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            device=config.audio.device,
            block_size=config.audio.block_size,
            queue_max_chunks=config.audio.queue_max_chunks,
        )

    def create_recognizer(
        self,
        config: AdapterConfigLike,
    ) -> StreamingRecognizer:
        """创建默认 FunASR 识别器。"""

        return StreamingRecognizer(config, self._config)


__all__ = ["FunASRProvider"]