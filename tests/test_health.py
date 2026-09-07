async def test_health_reports_every_component(client):
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "api": "ok",
        "postgres": "ok",
        "redis": "ok",
        "qdrant": "ok",
        "inference": "ok",
        "rabbitmq": "ok",
    }


async def test_health_is_degraded_not_failed_when_a_component_is_down(
    client,
    redis_client,
    inference,
    broker,
):
    redis_client.available = False
    inference.available = False
    broker.available = False

    response = await client.get("/health")
    body = response.json()

    # A partially broken stack must still answer 200 so the payload is readable.
    assert response.status_code == 200
    assert body["api"] == "ok"
    assert body["postgres"] == "ok"
    assert body["redis"] == "error"
    assert body["inference"] == "error"
    assert body["rabbitmq"] == "error"


async def test_worker_health_lists_responding_workers(client, broker):
    response = await client.get("/health/workers")

    assert response.status_code == 200
    assert response.json() == {"workers": ["celery@test"], "available": True}


async def test_worker_health_reports_an_empty_fleet(client, broker):
    broker.workers = []

    response = await client.get("/health/workers")

    assert response.status_code == 200
    assert response.json() == {"workers": [], "available": False}


async def test_metrics_endpoint_exposes_processing_counters(client):
    response = await client.get("/metrics")

    assert response.status_code == 200
    assert "documents_processing_total" in response.text
    assert "document_processing_duration_seconds" in response.text
