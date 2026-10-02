from nexus.transport import RequestLimits


def test_streamed_limit_without_content_length(app_factory, asgi_request):
    app = app_factory(limits=RequestLimits(max_body_bytes=10))
    chunks = [{"type": "http.request", "body": b"x" * 6, "more_body": True}] * 2
    assert asgi_request(app, chunks)[0]["status"] == 413


def test_slow_body_times_out(app_factory, asgi_request):
    app = app_factory(limits=RequestLimits(body_timeout_seconds=0.01))
    chunks = [{"type": "http.request", "body": b"{}", "more_body": False}]
    assert asgi_request(app, chunks, delay=0.1)[0]["status"] == 408


def test_disconnected_client(app_factory, asgi_request):
    assert asgi_request(app_factory(), [{"type": "http.disconnect"}]) == []


def test_streamed_valid_json(app_factory, asgi_request, payload):
    import json

    body = json.dumps(payload).encode()
    chunks = [
        {"type": "http.request", "body": body[:100], "more_body": True},
        {"type": "http.request", "body": body[100:], "more_body": False},
    ]
    assert asgi_request(app_factory(), chunks)[0]["status"] == 200
