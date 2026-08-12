"""FunASR ASR provider 插件配置。"""

from __future__ import annotations

from typing import ClassVar

from src.app.plugin_system.base import BaseConfig, Field, SectionBase, config_section


class FunASRProviderConfig(BaseConfig):
    """FunASR provider 配置。"""

    name: ClassVar[str] = "config"
    description: ClassVar[str] = "FunASR ASR provider 配置"

    @config_section("plugin", title="插件设置", tag="plugin")
    class PluginSection(SectionBase):
        """插件基本配置。"""

        enabled: bool = Field(
            default=True,
            description="是否启用 FunASR ASR provider",
            label="启用 Provider",
            tag="plugin",
        )
        register_as_default: bool = Field(
            default=True,
            description="注册后是否作为默认 ASR provider",
            label="设为默认 Provider",
            tag="plugin",
        )

    @config_section("asr", title="FunASR 模型", tag="ai")
    class AsrSection(SectionBase):
        """FunASR 模型与推理配置。"""

        model_type: str = Field(
            default="sensevoice_small",
            description="FunASR provider 使用的模型类型",
            label="模型类型",
            input_type="select",
            choices=["sensevoice_small", "funasr_nano"],
            tag="ai",
        )
        model: str = Field(
            default="FunAudioLLM/SenseVoiceSmall",
            description="FunASR 模型 ID 或本地模型目录",
            label="FunASR 模型",
            tag="ai",
        )
        hub: str = Field(
            default="ms",
            description="模型下载来源；SenseVoiceSmall 与 Fun-ASR-Nano 官方示例都可使用 hf 加载仓库中的 model.py",
            label="模型 Hub",
            input_type="select",
            choices=["hf", "ms"],
            tag="ai",
        )
        device: str = Field(
            default="auto",
            description="推理设备：auto 自动选择 GPU/CPU，cpu 使用 CPU，cuda:0/gpu 使用首张 GPU",
            label="推理设备",
            input_type="select",
            choices=["auto", "cpu", "cuda:0", "gpu"],
            tag="performance",
        )
        trust_remote_code: bool = Field(
            default=True,
            description="是否允许 FunASR 加载模型仓库中的远程 Python 代码；Fun-ASR-Nano 官方用法需要开启",
            label="信任远程代码",
            tag="ai",
        )
        remote_code: str = Field(
            default="./model.py",
            description="FunASR trust_remote_code=True 时导入的模型代码路径",
            label="远程代码模块",
            tag="ai",
        )
        vad_model: str = Field(
            default="fsmn-vad",
            description="FunASR VAD 模型；默认启用以按句切分整句识别",
            label="VAD 模型",
            tag="ai",
        )
        enable_streaming: bool = Field(
            default=False,
            description="是否启用流式中间推理；关闭后依赖 VAD/端点检测，仅在整句结束时输出最终文本",
            label="启用流式识别",
            tag="performance",
        )
        max_single_segment_time: int = Field(
            default=30000,
            description="FunASR VAD 单段最大时长，单位毫秒",
            label="最大分段时长(ms)",
            ge=1000,
            le=120000,
            tag="performance",
        )
        batch_size_s: int = Field(
            default=0,
            description="FunASR 动态 batch 秒数；Fun-ASR-Nano 官方示例使用 0",
            label="Batch 秒数",
            ge=0,
            le=300,
            tag="performance",
        )
        max_length: int = Field(
            default=32,
            description="Fun-ASR-Nano LLM 单次生成的最大 token 数；值越大延迟越高",
            label="最大生成 token",
            ge=16,
            le=512,
            tag="performance",
        )
        llm_dtype: str = Field(
            default="bf16",
            description="Fun-ASR-Nano LLM 推理精度；GPU 可尝试 bf16/fp16，CPU 建议 fp32",
            label="LLM 精度",
            input_type="select",
            choices=["bf16", "fp16", "fp32"],
            tag="performance",
        )
        ctc_only: bool = Field(
            default=False,
            description="仅对 Fun-ASR-Nano 生效；跳过 LLM 生成，仅返回 CTC 快速识别文本",
            label="仅使用 CTC",
            tag="performance",
        )
        inference_window_ms: int = Field(
            default=1600,
            description="流式识别开启时，累计多少毫秒音频后执行一次中间推理",
            label="推理窗口(ms)",
            ge=200,
            le=10000,
            tag="performance",
        )
        max_audio_window_ms: int = Field(
            default=30000,
            description="保留给一次识别的最大音频窗口，超过后丢弃更早音频以控制延迟和显存",
            label="最大音频窗口(ms)",
            ge=1000,
            le=60000,
            tag="performance",
        )
        endpoint_silence_ms: int = Field(
            default=900,
            description="简易端点检测的尾部静音毫秒数",
            label="端点静音(ms)",
            ge=200,
            le=5000,
            tag="performance",
        )
        endpoint_rms_threshold: float = Field(
            default=0.008,
            description="简易端点检测的 RMS 静音阈值",
            label="端点 RMS",
            ge=0.0,
            le=1.0,
            tag="performance",
        )
        enable_endpoint: bool = Field(
            default=True,
            description="是否启用端点检测，开启后会按语音停顿提交最终文本",
            label="启用端点检测",
            tag="performance",
        )

    plugin: PluginSection = Field(default_factory=PluginSection)
    asr: AsrSection = Field(default_factory=AsrSection)


__all__ = ["FunASRProviderConfig"]