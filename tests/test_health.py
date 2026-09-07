async def test_health_reports_every_component(client):
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "api": "ok",
        "postgres": "ok",
        "redis": "ok",
        "qdrant": "ok",
        "inference": "ok",
    }


async def test_health_is_degraded_not_failed_when_a_component_is_down(
    client,
    redis_client,
    inference,
):
    redis_client.available = False
    inference.available = False

    response = await client.get("/health")
    body = response.json()

    # A partially broken stack must still answer 200 so the payload is readable.
    assert response.status_code == 200
    assert body["api"] == "ok"
    assert body["postgres"] == "ok"
    assert body["redis"] == "error"
    assert body["inference"] == "error"
