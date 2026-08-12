"""FunASR Provider 的注册生命周期与无硬件辅助逻辑测试。"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plugins.funasr_asr_provider_anima.config import FunASRProviderConfig  # noqa: E402
from plugins.funasr_asr_provider_anima.plugin import (  # noqa: E402
    FunASRASRProviderPlugin,
)
from plugins.funasr_asr_provider_anima.provider import FunASRProvider  # noqa: E402
from plugins.funasr_asr_provider_anima.src.audio_source import (  # noqa: E402
    MicrophoneAudioSource,
)


class _Registry:
    """记录 Provider 注册与注销调用的测试 Registry。"""

    def __init__(self, *, register_error: Exception | None = None) -> None:
        """保存可选注册异常。"""

        self.register_error = register_error
        self.registered: list[tuple[Any, bool]] = []
        self.unregistered: list[str] = []

    def register_provider(self, provider: Any, *, default: bool = False) -> None:
        """记录注册或抛出配置异常。"""

        if self.register_error is not None:
            raise self.register_error
        self.registered.append((provider, default))

    def unregister_provider(self, provider_name: str) -> None:
        """记录注销名称。"""

        self.unregistered.append(provider_name)


def test_audio_device_coercion_supports_negative_indices() -> None:
    """设备配置应支持空值、正负索引和设备名称。"""

    assert MicrophoneAudioSource._coerce_device(None) is None
    assert MicrophoneAudioSource._coerce_device("") is None
    assert MicrophoneAudioSource._coerce_device(" 3 ") == 3
    assert MicrophoneAudioSource._coerce_device("-1") == -1
    assert MicrophoneAudioSource._coerce_device("Virtual Cable") == "Virtual Cable"


@pytest.mark.asyncio
async def test_plugin_registers_and_unregisters_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """启用时应注册 FunASR Provider，并在卸载时按名称注销。"""

    from plugins.funasr_asr_provider_anima import plugin as plugin_module

    registry = _Registry()
    monkeypatch.setattr(plugin_module, "get_service", lambda _signature: registry)
    config = FunASRProviderConfig()
    config.plugin.enabled = True
    config.plugin.register_as_default = True
    plugin = FunASRASRProviderPlugin(config)

    await plugin.on_plugin_loaded()

    assert isinstance(plugin.provider, FunASRProvider)
    assert registry.registered == [(plugin.provider, True)]

    await plugin.on_plugin_unloaded()

    assert registry.unregistered == [FunASRProvider.provider_name]
    assert plugin.provider is None


@pytest.mark.asyncio
async def test_plugin_missing_registry_raises_without_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry 缺失时加载应失败且不保留 Provider 引用。"""

    from plugins.funasr_asr_provider_anima import plugin as plugin_module

    monkeypatch.setattr(plugin_module, "get_service", lambda _signature: None)
    plugin = FunASRASRProviderPlugin(FunASRProviderConfig())

    with pytest.raises(RuntimeError, match="无法获取 asr_adapter_anima"):
        await plugin.on_plugin_loaded()

    assert plugin.provider is None


@pytest.mark.asyncio
async def test_registration_failure_rolls_back_provider_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry 注册异常时不得留下半注册 Provider 引用。"""

    from plugins.funasr_asr_provider_anima import plugin as plugin_module

    registry = _Registry(register_error=RuntimeError("registry unavailable"))
    monkeypatch.setattr(plugin_module, "get_service", lambda _signature: registry)
    plugin = FunASRASRProviderPlugin(FunASRProviderConfig())

    with pytest.raises(RuntimeError, match="registry unavailable"):
        await plugin.on_plugin_loaded()

    assert plugin.provider is None


@pytest.mark.asyncio
async def test_disabled_plugin_skips_registry_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """禁用配置应直接跳过 Registry 访问。"""

    from plugins.funasr_asr_provider_anima import plugin as plugin_module

    looked_up = False

    def _get_service(_signature: str) -> None:
        """记录不应发生的 Service 查询。"""

        nonlocal looked_up
        looked_up = True
        return None

    monkeypatch.setattr(plugin_module, "get_service", _get_service)
    config = FunASRProviderConfig()
    config.plugin.enabled = False
    plugin = FunASRASRProviderPlugin(config)

    await plugin.on_plugin_loaded()

    assert looked_up is False
    assert plugin.provider is None
