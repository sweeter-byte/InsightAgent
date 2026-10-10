"""Lightweight readiness checks over already-owned application resources."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict


class AsyncPing(Protocol):
    async def ping(self) -> Any: ...


class ReadyProbe(Protocol):
    def check_ready(self) -> None: ...


class ReadinessStatus(str, Enum):
    READY = "ready"
    NOT_READY = "not_ready"


class ComponentStatus(str, Enum):
    READY = "ready"
    UNAVAILABLE = "unavailable"
    AVAILABLE = "available"
    UNCONFIGURED = "unconfigured"


class ComponentReadiness(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: ComponentStatus
    detail: str | None = None


class ReadinessResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: ReadinessStatus
    required: dict[str, ComponentReadiness]
    optional: dict[str, ComponentReadiness]


class ReadinessService:
    """Check mandatory local resources without running model/provider work."""

    def __init__(
        self,
        *,
        redis: AsyncPing,
        qdrant: ReadyProbe,
        checkpoint: ReadyProbe,
        web_available: bool,
        vision_available: bool,
    ) -> None:
        self.redis = redis
        self.qdrant = qdrant
        self.checkpoint = checkpoint
        self.web_available = web_available
        self.vision_available = vision_available

    async def check(self) -> ReadinessResult:
        required = {
            "redis": await self._check_redis(),
            "qdrant": self._check_sync("qdrant", self.qdrant),
            "checkpoint": self._check_sync("checkpoint", self.checkpoint),
            "config": ComponentReadiness(status=ComponentStatus.READY),
        }
        ready = all(
            component.status is ComponentStatus.READY
            for component in required.values()
        )
        return ReadinessResult(
            status=ReadinessStatus.READY if ready else ReadinessStatus.NOT_READY,
            required=required,
            optional={
                "web": self._optional(self.web_available),
                "vision": self._optional(self.vision_available),
            },
        )

    async def _check_redis(self) -> ComponentReadiness:
        try:
            if not await self.redis.ping():
                raise RuntimeError("redis ping returned false")
        except Exception:
            return ComponentReadiness(
                status=ComponentStatus.UNAVAILABLE,
                detail="redis unavailable",
            )
        return ComponentReadiness(status=ComponentStatus.READY)

    @staticmethod
    def _check_sync(name: str, probe: ReadyProbe) -> ComponentReadiness:
        try:
            probe.check_ready()
        except Exception:
            return ComponentReadiness(
                status=ComponentStatus.UNAVAILABLE,
                detail=f"{name} unavailable",
            )
        return ComponentReadiness(status=ComponentStatus.READY)

    @staticmethod
    def _optional(available: bool) -> ComponentReadiness:
        return ComponentReadiness(
            status=(
                ComponentStatus.AVAILABLE
                if available
                else ComponentStatus.UNCONFIGURED
            )
        )
