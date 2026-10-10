import threading
from pathlib import Path
from unittest.mock import patch

from client import AntigravityClient


def test_prewarmer_starts_selected_model_before_first_request():
    with AntigravityClient() as client:
        started = threading.Event()
        calls = []

        def fake_spawn(model, effort):
            calls.append((model, effort))
            started.set()

        with patch('client.is_authenticated', return_value=True), patch.object(client, '_get_or_spawn_worker', side_effect=fake_spawn):
            client.start_prewarmer(model='gemini-3.8-flash', reasoning_config={'effort': 'low'})
            assert started.wait(5)
        assert calls == [('gemini-3.8-flash-low', 'low')]


def test_closed_client_cannot_spawn_worker_after_delayed_prewarm():
    with AntigravityClient() as client:
        waiting = threading.Event()
        resume = threading.Event()
        spawned = []

        def slow_auth():
            waiting.set()
            assert resume.wait(5)
            return True

        with patch('client.is_authenticated', side_effect=slow_auth), patch.object(client, '_get_or_spawn_worker', side_effect=lambda *a: spawned.append(a)):
            client.start_prewarmer(model='gemini-3.8-flash')
            assert waiting.wait(5)
            client.close()
            resume.set()
            client._prewarm_thread.join(5)
        assert not client._prewarm_thread.is_alive()
        assert not spawned
        assert not Path(client._cwd).exists()
