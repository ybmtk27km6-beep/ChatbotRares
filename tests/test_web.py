"""Focused API tests for storage, uploads, provider selection, and SSE."""

from __future__ import annotations

import io
import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import app as web


class WebApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.history_root = web.HISTORY_ROOT
        web.HISTORY_ROOT = web.Path(self.temp_dir.name)
        self.client = web.create_app({"TESTING": True, "SECRET_KEY": "test-key"}).test_client()

    def tearDown(self) -> None:
        web.HISTORY_ROOT = self.history_root
        self.temp_dir.cleanup()

    def test_index_sets_browser_session_and_serves_app_shell(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"AI STUDIO", response.data)
        self.assertIn(b"manifest.json", response.data)
        self.assertIn(b"katex", response.data.lower())

    def test_provider_auto_prefers_cloud_when_key_exists(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AI_PROVIDER": "auto",
                "CLOUD_API_KEY": "test-token",
                "CLOUD_BASE_URL": "https://api.groq.com/openai/v1",
                "CLOUD_MODEL": "test-model",
            },
            clear=False,
        ):
            provider, model, target = web._active_provider()
        self.assertEqual(provider, "cloud")
        self.assertEqual(model, "test-model")
        self.assertTrue(target.endswith("|test-token"))

    def test_local_provider_defaults_to_qwen(self) -> None:
        with patch.dict(os.environ, {"AI_PROVIDER": "local"}, clear=True):
            _, model, _ = web._active_provider()
        self.assertEqual(model, "qwen2.5:14b")

    def test_stream_saves_history_and_emits_sse_events(self) -> None:
        chat_id = "a" * 32

        def fake_stream(_model: str, _messages: list[dict[str, object]]):
            self.assertIn("single dollar signs ($...$)", _messages[0]["content"])
            self.assertIn("double dollar signs ($$...$$)", _messages[0]["content"])
            yield "Hello "
            yield "local."

        with patch.dict(
            os.environ,
            {"AI_PROVIDER": "local", "OLLAMA_MODEL": "llama3.2"},
            clear=False,
        ), patch.object(web, "_stream_local", fake_stream):
            response = self.client.post(
                "/api/chat/stream",
                json={
                    "conversation_id": chat_id,
                    "message": "Say hello",
                    "display_message": "Say hello\n📄 notes.txt",
                    "persona": "Academic Tutor",
                    "style": "steps",
                },
                buffered=True,
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"event: token", response.data)
        self.assertIn(b"event: done", response.data)
        record = json.loads((web.HISTORY_ROOT / self._owner_id() / f"{chat_id}.json").read_text())
        self.assertEqual(len(record["messages"]), 2)
        self.assertEqual(record["messages"][1]["content"], "Hello local.")
        self.assertEqual(record["messages"][0]["display_content"], "Say hello\n📄 notes.txt")
        self.assertEqual(record["style"], "steps")

    def test_katex_assets_are_allowed_by_content_security_policy(self) -> None:
        response = self.client.get("/")
        policy = response.headers["Content-Security-Policy"]
        self.assertIn("https://cdn.jsdelivr.net", policy.split("script-src ", 1)[1].split(";", 1)[0])
        self.assertIn("https://cdn.jsdelivr.net", policy.split("style-src ", 1)[1].split(";", 1)[0])
        self.assertIn("https://cdn.jsdelivr.net", policy.split("font-src ", 1)[1].split(";", 1)[0])

    def test_upload_extracts_text_without_persisting_upload(self) -> None:
        response = self.client.post(
            "/upload",
            data={"files": (io.BytesIO(b"private notes"), "notes.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["files"][0]["chunks"], ["private notes"])
        self.assertEqual(response.json["files"][0]["kind"], "document")
        self.assertEqual(list(web.HISTORY_ROOT.rglob("*.txt")), [])

    def test_upload_accepts_images_and_rejects_mismatched_signatures(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
        response = self.client.post(
            "/upload",
            data={"files": (io.BytesIO(png), "diagram.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        image = response.json["files"][0]
        self.assertEqual(image["kind"], "image")
        self.assertEqual(image["mime_type"], "image/png")
        self.assertEqual(image["data"], web.base64.b64encode(png).decode("ascii"))

        invalid = self.client.post(
            "/upload",
            data={"files": (io.BytesIO(b"not an image"), "diagram.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(invalid.status_code, 400)

    def test_local_image_chat_routes_to_vision_model(self) -> None:
        received: dict[str, object] = {}

        def fake_stream(model: str, messages: list[dict[str, object]]):
            received["model"] = model
            received["messages"] = messages
            yield "A diagram."

        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
        with patch.dict(
            os.environ,
            {
                "AI_PROVIDER": "local",
                "OLLAMA_MODEL": "qwen2.5:14b",
                "OLLAMA_VISION_MODEL": "qwen2.5vl:7b",
            },
            clear=False,
        ), patch.object(web, "_stream_local", fake_stream):
            response = self.client.post(
                "/api/chat/stream",
                json={
                    "conversation_id": "b" * 32,
                    "message": "Read the diagram",
                    "images": [
                        {
                            "name": "diagram.png",
                            "mime_type": "image/png",
                            "data": web.base64.b64encode(png).decode("ascii"),
                        }
                    ],
                },
                buffered=True,
            )
        self.assertIn(b"event: done", response.data)
        self.assertEqual(received["model"], "qwen2.5vl:7b")
        messages = received["messages"]
        self.assertEqual(messages[-1]["images"][0]["name"], "diagram.png")

    def test_cloud_stream_encodes_images_as_openai_compatible_content(self) -> None:
        calls: dict[str, object] = {}

        class FakeCompletions:
            def create(self, **kwargs: object):
                calls.update(kwargs)
                return iter(
                    [
                        SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    delta=SimpleNamespace(content="seen")
                                )
                            ]
                        )
                    ]
                )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        messages = [
            {
                "role": "user",
                "content": "Describe it",
                "images": [
                    {
                        "mime_type": "image/png",
                        "data": "cG5n",
                    }
                ],
            }
        ]
        with patch.object(web, "_cloud_client", return_value=client):
            self.assertEqual(
                list(web._stream_cloud("vision-model", "https://api.example|key", messages)),
                ["seen"],
            )
        provider_message = calls["messages"][0]
        self.assertNotIn("images", provider_message)
        self.assertEqual(
            {key: value for key, value in calls.items() if key != "messages"},
            {
                "model": "vision-model",
                "temperature": 0.3,
                "stream": True,
            },
        )
        self.assertEqual(
            provider_message["content"][1]["image_url"]["url"],
            "data:image/png;base64,cG5n",
        )

    def test_cloud_stream_discards_malformed_messages(self) -> None:
        calls: dict[str, object] = {}

        class FakeCompletions:
            def create(self, **kwargs: object):
                calls.update(kwargs)
                return iter([])

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        messages = [
            {"role": "developer", "content": "unsupported"},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "   "},
            {"role": "user", "content": 123},
            {"role": "system", "content": "Instructions"},
            {"role": "user", "content": "Hello"},
        ]
        with patch.object(web, "_cloud_client", return_value=client):
            self.assertEqual(
                list(web._stream_cloud("test-model", "https://api.example|key", messages)),
                [],
            )
        self.assertEqual(
            calls["messages"],
            [
                {"role": "system", "content": "Instructions"},
                {"role": "user", "content": "Hello"},
            ],
        )

    def test_cloud_status_filters_non_chat_models(self) -> None:
        models_response = SimpleNamespace(
            data=[
                SimpleNamespace(id="llama-3.3-70b-versatile"),
                SimpleNamespace(id="whisper-large-v3"),
                SimpleNamespace(id="orpheus-3b"),
                SimpleNamespace(id="prompt-guard-2"),
                SimpleNamespace(id="llama-guard-safeguard"),
                SimpleNamespace(id="llama-3.1-8b-instant"),
            ]
        )
        client = SimpleNamespace(
            models=SimpleNamespace(list=lambda: models_response)
        )
        with patch.dict(
            os.environ,
            {
                "AI_PROVIDER": "cloud",
                "CLOUD_API_KEY": "test-key",
                "CLOUD_MODEL": "whisper-large-v3",
            },
            clear=False,
        ), patch.object(web, "_cloud_client", return_value=client):
            response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["model"], "llama-3.3-70b-versatile")
        self.assertEqual(
            response.json["models"],
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        )

    def test_cloud_stream_error_is_logged_and_sent_to_the_ui(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AI_PROVIDER": "cloud",
                "CLOUD_API_KEY": "test-key",
                "CLOUD_MODEL": "llama-3.3-70b-versatile",
            },
            clear=False,
        ), patch.object(
            web, "_stream_cloud", side_effect=RuntimeError("model is unavailable")
        ), patch("builtins.print") as print_mock:
            response = self.client.post(
                "/api/chat/stream",
                json={"message": "Hello"},
                buffered=True,
            )
        self.assertIn(b"model is unavailable", response.data)
        self.assertEqual(
            print_mock.call_args_list[0].args,
            ("GROQ ERROR:", "RuntimeError('model is unavailable')"),
        )
        self.assertEqual(
            print_mock.call_args_list[0].kwargs,
            {"flush": True},
        )

    def test_restricted_python_runner_and_public_mode_block(self) -> None:
        with patch.dict(os.environ, {"AI_PROVIDER": "local", "APP_MODE": "local"}, clear=False):
            result = self.client.post(
                "/api/run-code",
                json={"code": "total = sum(range(4))\nprint(total)"},
            )
            blocked_syntax = self.client.post(
                "/api/run-code",
                json={"code": "import os\nprint(os.name)"},
            )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json["output"], "6")
        self.assertEqual(blocked_syntax.status_code, 400)

        with patch.dict(os.environ, {"APP_MODE": "public"}, clear=False):
            public_response = self.client.post(
                "/api/run-code",
                json={"code": "print('not allowed')"},
            )
        self.assertEqual(public_response.status_code, 403)

        with patch.dict(
            os.environ,
            {
                "AI_PROVIDER": "cloud",
                "CLOUD_API_KEY": "test-key",
                "APP_MODE": "local",
            },
            clear=False,
        ):
            cloud_response = self.client.post(
                "/api/run-code",
                json={"code": "print('not allowed')"},
            )
        self.assertEqual(cloud_response.status_code, 403)

    def test_explicit_cloud_mode_without_key_does_not_enable_runner(self) -> None:
        with patch.dict(
            os.environ,
            {"AI_PROVIDER": "cloud", "APP_MODE": "local"},
            clear=True,
        ):
            response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["online"])
        self.assertEqual(response.json["provider"], "cloud")
        self.assertFalse(response.json["execution_enabled"])

    def test_web_search_sources_are_streamed_and_saved(self) -> None:
        chat_id = "c" * 32

        def fake_stream(_model: str, messages: list[dict[str, object]]):
            self.assertIn("Web search results", messages[-1]["content"])
            yield "Grounded answer."

        sources = [
            {
                "title": "Source",
                "url": "https://example.com/",
                "snippet": "A verified snippet.",
            }
        ]
        with patch.dict(os.environ, {"AI_PROVIDER": "local"}, clear=False), patch.object(
            web, "_web_search", return_value=sources
        ), patch.object(web, "_stream_local", fake_stream):
            response = self.client.post(
                "/api/chat/stream",
                json={
                    "conversation_id": chat_id,
                    "message": "Find a source",
                    "web_search": True,
                    "search_query": "source",
                },
                buffered=True,
            )
        self.assertIn(b"event: sources", response.data)
        self.assertIn(b"event: done", response.data)
        record = json.loads(
            (web.HISTORY_ROOT / self._owner_id() / f"{chat_id}.json").read_text()
        )
        self.assertEqual(record["messages"][1]["sources"], sources)

    def test_local_ollama_stream_uses_requested_runtime_options(self) -> None:
        calls: dict[str, object] = {}

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                calls["client"] = kwargs

            def chat(self, **kwargs: object):
                calls["chat"] = kwargs
                return iter([{"message": {"content": "ok"}}])

        with patch.dict(os.environ, {"OLLAMA_HOST": "http://localhost:11434"}), patch.object(
            web.ollama, "Client", FakeClient
        ):
            self.assertEqual(
                list(web._stream_local("qwen2.5:14b", [{"role": "user", "content": "hi"}])),
                ["ok"],
            )
        call = calls["chat"]
        self.assertEqual(call["keep_alive"], "30m")
        self.assertEqual(
            call["options"],
            {"num_ctx": 4096, "temperature": 0.3, "repeat_penalty": 1.15},
        )

    def test_rejects_bad_conversation_id(self) -> None:
        response = self.client.get("/api/chats/../etc")
        self.assertNotEqual(response.status_code, 200)

    def _owner_id(self) -> str:
        with self.client.session_transaction() as browser_session:
            return browser_session["owner_id"]


if __name__ == "__main__":
    unittest.main()
