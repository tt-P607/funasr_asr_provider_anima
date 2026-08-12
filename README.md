# funasr_asr_provider_anima (言柒 Fork 版)

> [!IMPORTANT]
> **关键依赖声明**：本插件是言柒针对 `tt-P607/anima_chatter` 及其配套的 `tt-P607/asr_adapter_anima` 定制的专属 FunASR 后端插件。如果您要使用言柒版本的实时语音通话全套方案，**必须**使用本插件作为 ASR Provider 后端，以保证正常与 `asr_adapter_anima` 进行协作。

## 部署与安装

要完整搭建言柒版 `anima_chatter` 双向实时语音通话环境，请在 `plugins/` 目录下克隆全套配套插件：

```bash
# 1. 克隆语音/VTS 主聊天插件 (Anima Chatter)
git clone https://github.com/tt-P607/anima_chatter.git

# 2. 克隆配套实时语音识别适配器 (ASR Adapter Anima)
git clone https://github.com/tt-P607/asr_adapter_anima.git

# 3. 克隆本 FunASR 识别后端 (FunASR Provider Anima)
git clone https://github.com/tt-P607/funasr_asr_provider_anima.git
```

## 概述

`funasr_asr_provider_anima` 是独立的 FunASR 识别后端插件。

它不直接承担平台适配职责，而是在加载时向 `asr_adapter_anima` 提供的 provider registry 注册 `funasr` 识别后端。这样 `asr_adapter_anima` 可以继续保持为通用本地语音 Adapter，而具体模型和推理配置都收敛在本插件内。

## 提供的能力

该插件本身不额外暴露公开组件；它的主要职责是：

- 在 `on_plugin_loaded()` 时注册 `funasr` provider
- 在 `on_plugin_unloaded()` 时注销 `funasr` provider
- 提供基于 FunASR 的识别器与麦克风音频源实现
- 持有 FunASR 专属模型和推理配置

provider 名称：

- `funasr`

## 依赖

插件依赖：

- `asr_adapter_anima`
- `asr_adapter_anima:service:asr_provider_registry`

Python 依赖较重，主要包括：

- `funasr`
- `torch`
- `torchaudio`
- `transformers`
- `tokenizers`
- `modelscope`
- `huggingface-hub`
- `sounddevice`
- `numpy`

## 配置

配置文件路径：

- `config/plugins/funasr_asr_provider_anima/config.toml`

主要配置节：

- `plugin`：是否启用、是否注册为默认 provider
- `asr`：模型 ID、hub、设备、远程代码、VAD、是否启用流式识别、推理窗口、音频窗口、端点检测、CTC-only 等 FunASR 专属参数

当前默认值：

- 默认模型为 `FunAudioLLM/SenseVoiceSmall`
- 默认启用 `fsmn-vad` 做整句切分
- 默认关闭流式识别，只在整句结束时输出最终文本

这部分配置不应再写在 `asr_adapter_anima` 中。

## 工作方式

1. 插件加载后，向 `asr_adapter_anima` 的 registry 注册 `FunASRProvider`。
2. `asr_adapter_anima` 根据自己的 `config.asr.provider` 选择 `funasr`。
3. provider 使用 adapter 的通用音频配置创建音频源。
4. provider 使用自己的 FunASR 配置创建识别器。
5. 识别器负责模型初始化、remote code 解析、端点检测和结果提取。

## 适用场景

- 想把本地实时麦克风识别与具体模型实现解耦
- 想在不改 `asr_adapter_anima` 的情况下替换或并存多个 ASR 后端
- 想把 FunASR 的重依赖和模型配置独立管理

## 故障排查

- 模型文件校验失败：确认 `asr.model` 指向可访问的 Hub ID 或完整本地目录。
- remote code 导入失败：确认 `trust_remote_code`、`remote_code` 与所选 Hub/模型仓库一致；本地路径会自动加入导入搜索路径且避免重复注入。
- CUDA 不可用或显存不足：将 `asr.device` 改为 `cpu`，并根据模型调整精度、窗口和生成长度。
- 输入设备无法打开：设备留空使用系统默认设备，也可填写设备名称、正索引或负索引。
- 插件加载失败后 Registry 存在残留：当前实现只在注册成功后保存 Provider 引用，失败会完整回滚。

## 验证

自动测试位于 `plugins/funasr_asr_provider_anima/test/`：

```bash
uv run pytest plugins/funasr_asr_provider_anima/test -q --no-cov
uv run ruff check plugins/funasr_asr_provider_anima
```

自动测试不加载 FunASR 模型。实机验证需额外确认模型下载/本地加载、选定设备推理、麦克风采集、端点检测和最终文本输出。

## 相关插件

- `plugins/asr_adapter_anima`
- `plugins/anima_chatter`
