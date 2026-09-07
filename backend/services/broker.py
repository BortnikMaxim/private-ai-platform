"""Liveness probes for RabbitMQ and the Celery worker fleet.

Kept behind a small class so ``/health`` can be faked in tests exactly like the
other backing services.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


class BrokerClient:
    def __init__(
        self,
        broker_url: str,
        health_timeout: float = 2.0,
        ping_timeout: float = 1.0,
    ) -> None:
        self.broker_url = broker_url
        self.health_timeout = health_timeout
        self.ping_timeout = ping_timeout

    async def health(self) -> bool:
        """Open an AMQP connection and drop it again. Fast and non-blocking."""

        def probe() -> bool:
            import kombu

            with kombu.Connection(
                self.broker_url,
                connect_timeout=self.health_timeout,
            ) as connection:
                connection.ensure_connection(max_retries=0, timeout=self.health_timeout)
            return True

        try:
            return await asyncio.to_thread(probe)
        except Exception as exc:  # noqa: BLE001 - health checks never raise
            logger.warning("rabbitmq_health_failed error=%s", type(exc).__name__)
            return False

    async def ping_workers(self) -> list[str]:
        """Names of the workers that answer a broadcast ping.

        Slower than :meth:`health` (it waits out the timeout when nobody
        answers), which is why it is not part of ``GET /health``.
        """

        def probe() -> list[str]:
            from backend.worker.celery_app import celery_app

            replies = celery_app.control.ping(timeout=self.ping_timeout) or []
            return [name for reply in replies for name in reply]

        try:
            return await asyncio.to_thread(probe)
        except Exception as exc:  # noqa: BLE001
            logger.warning("celery_ping_failed error=%s", type(exc).__name__)
            return []
