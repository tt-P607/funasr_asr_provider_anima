"""FunASR ASR provider 插件入口。"""

from __future__ import annotations

from typing import cast

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.service_api import get_service
from src.app.plugin_system.base import BasePlugin, register_plugin

from .config import FunASRProviderConfig
from .protocol import ASRProviderRegistryLike
from .provider import FunASRProvider


logger = get_logger("funasr_asr_provider")


@register_plugin
class FunASRASRProviderPlugin(BasePlugin):
    """注册并管理 Anima 套件使用的 FunASR Provider。"""

    plugin_name = "funasr_asr_provider_anima"
    configs = [FunASRProviderConfig]
    dependent_components = ["asr_adapter_anima:service:asr_provider_registry"]

    def __init__(self, config: FunASRProviderConfig | None = None) -> None:
        """初始化插件。"""

        super().__init__(config)
        self.provider: FunASRProvider | None = None

    def get_components(self) -> list[type]:
        """provider 插件不额外暴露组件。"""

        return []

    async def on_plugin_loaded(self) -> None:
        """向 asr_adapter 注册 FunASR provider。"""

        config = (
            self.config
            if isinstance(self.config, FunASRProviderConfig)
            else FunASRProviderConfig()
        )
        if not config.plugin.enabled:
            logger.info("FunASR ASR provider 已禁用")
            return

        registry = cast(
            ASRProviderRegistryLike | None,
            get_service("asr_adapter_anima:service:asr_provider_registry"),
        )
        if registry is None:
            raise RuntimeError("无法获取 asr_adapter_anima provider registry service")

        provider = FunASRProvider(config)
        try:
            registry.register_provider(
                provider,
                default=config.plugin.register_as_default,
            )
        except Exception:
            self.provider = None
            raise
        self.provider = provider
        logger.info("FunASR ASR Provider 已注册")

    async def on_plugin_unloaded(self) -> None:
        """注销 FunASR provider。"""

        registry = cast(
            ASRProviderRegistryLike | None,
            get_service("asr_adapter_anima:service:asr_provider_registry"),
        )
        if registry is not None:
            registry.unregister_provider(FunASRProvider.provider_name)
        self.provider = None
        logger.info("FunASR ASR Provider 已注销")


__all__ = ["FunASRASRProviderPlugin"]