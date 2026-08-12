"""本机麦克风音频采集封装。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.app.plugin_system.api.log_api import get_logger


logger = get_logger("funasr_asr_provider.audio")


class MicrophoneAudioSource:
    """使用 sounddevice 采集本机麦克风 float32 单声道音频。"""

    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        device: str,
        block_size: int,
        queue_max_chunks: int,
    ) -> None:
        """初始化音频源。"""

        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device.strip() or None
        self.block_size = block_size
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=queue_max_chunks)
        self._stream: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._dropped_chunks = 0
        self._last_status_warning_at = 0.0

    async def start(self) -> None:
        """启动麦克风采集。"""

        if self._running:
            return

        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("缺少依赖 sounddevice，请先安装插件依赖") from exc

        self._loop = asyncio.get_running_loop()
        device = self._coerce_device(self.device)
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.block_size,
            device=device,
            callback=self._on_audio,
        )
        self._stream.start()
        self._running = True
        actual_rate = getattr(self._stream, "samplerate", self.sample_rate)
        if int(actual_rate) != int(self.sample_rate):
            logger.warning(
                f"麦克风实际采样率与配置不一致: requested={self.sample_rate}, actual={actual_rate}"
            )

    async def stop(self) -> None:
        """停止麦克风采集并清空缓冲队列。"""

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False
        while not self._queue.empty():
            self._queue.get_nowait()

    async def read(self) -> Any:
        """读取一个音频块。"""

        return await self._queue.get()

    def drain_pending(self) -> list[Any]:
        """非阻塞取出当前已积压的音频块。"""

        chunks: list[Any] = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                return chunks

    def is_running(self) -> bool:
        """返回音频源是否正在运行。"""

        return self._running

    def queue_size(self) -> int:
        """返回当前等待处理的音频块数量。"""

        return self._queue.qsize()

    def queue_capacity(self) -> int:
        """返回音频队列最大容量。"""

        return self._queue.maxsize

    def _on_audio(self, indata: Any, _frames: int, _time: Any, status: Any) -> None:
        """sounddevice 回调：只做线程安全入队。"""

        if status:
            now = time.monotonic()
            if self._loop is not None and now - self._last_status_warning_at >= 2.0:
                self._last_status_warning_at = now
                self._loop.call_soon_threadsafe(logger.warning, f"麦克风输入状态异常: {status}")
        if self._loop is None or not self._running:
            return

        samples = self._to_mono(indata)
        self._loop.call_soon_threadsafe(self._put_latest, samples.copy())

    def _put_latest(self, samples: Any) -> None:
        """放入最新音频块；队列满时丢弃最旧块以保持实时性。"""

        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._dropped_chunks += 1
                if self._dropped_chunks == 1 or self._dropped_chunks % 20 == 0:
                    logger.warning(f"ASR 音频队列已满，累计丢弃旧音频块: {self._dropped_chunks}")
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(samples)

    def _to_mono(self, samples: Any) -> Any:
        """将 sounddevice 输入转换为一维 float32 单声道数组。"""

        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("缺少依赖 numpy，请先安装插件依赖") from exc

        array = np.asarray(samples, dtype=np.float32)
        if array.ndim == 2:
            if array.shape[1] == 1:
                return array[:, 0]
            return array.mean(axis=1, dtype=np.float32)
        return array.reshape(-1)

    @staticmethod
    def _coerce_device(device: str | None) -> str | int | None:
        """将配置中的设备值转换为 sounddevice 可接受的类型。"""

        value = str(device or "").strip()
        if not value:
            return None
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
        return value