"""FunASR ASR provider 插件入口。"""

from __future__ import annotations

from typing import cast

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.service_api import get_service
from src.core.components.base.plugin import BasePlugin
from src.core.components.loader import register_plugin

from .config import FunASRProviderConfig
from .protocol import ASRProviderRegistryLike
from .provider import FunASRProvider


logger = get_logger("funasr_asr_provider")


@register_plugin
class FunASRASRProviderPlugin(BasePlugin):
    """独立的 FunASR ASR provider 插件。"""

    plugin_name = "funasr_asr_provider_anima"
    plugin_version = "1.0.0"
    plugin_description = "FunASR ASR provider"
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

        config = self.config if isinstance(self.config, FunASRProviderConfig) else FunASRProviderConfig()
        if not config.plugin.enabled:
            logger.info("FunASR ASR provider 已禁用")
            return

        registry = cast(
            ASRProviderRegistryLike | None,
            get_service("asr_adapter_anima:service:asr_provider_registry"),
        )
        if registry is None:
            raise RuntimeError("无法获取 asr_adapter_anima provider registry service")

        self.provider = FunASRProvider(config)
        registry.register_provider(
            self.provider,
            default=config.plugin.register_as_default,
        )
        logger.info("FunASR ASR provider 已注册")

    async def on_plugin_unloaded(self) -> None:
        """注销 FunASR provider。"""

        registry = cast(
            ASRProviderRegistryLike | None,
            get_service("asr_adapter_anima:service:asr_provider_registry"),
        )
        if registry is not None:
            registry.unregister_provider(FunASRProvider.provider_name)
        self.provider = None


__all__ = ["FunASRASRProviderPlugin"]