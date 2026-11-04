"""Settings: defaults, file, preset, environment, flags; coercion; errors; the printed table."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stranger.config import ConfigError, Settings, coerce, format_settings, load_settings

FILE = '''
[model]
model = "qwen3:1.7b"
temperature = 0.5
stop = ["A:", "B:"]
[agent]
user_name = "Visitor"
[speech]
device = 7
[presets.cluster.model]
host = "http://localhost:8000"
api = "openai"
num_ctx = 32768
[presets.quiet.web]
quiet = true
'''


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "stranger.toml"
        self.path.write_text(FILE)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_defaults_when_no_file(self) -> None:
        s, src = load_settings(env={}, config_path=None)
        # no file in the temp env: cwd may still hold the repo's stranger.toml, so only check the shape
        self.assertEqual(set(src) >= {"model.model", "agent.user_name", "web.port"}, True)
        self.assertEqual(Settings().agent.user_name, "Visitor")
        self.assertEqual(Settings().web.port, 8766)

    def test_file_values_and_sources(self) -> None:
        s, src = load_settings(self.path, env={})
        self.assertEqual(s.source_file, self.path.resolve())
        self.assertEqual((s.model.temperature, s.model.stop, s.speech.device), (0.5, ("A:", "B:"), 7))
        self.assertEqual((src["model.temperature"], src["model.host"]), ("file", "default"))

    def test_preset_overlays_a_section(self) -> None:
        s, src = load_settings(self.path, preset="cluster", env={})
        self.assertEqual((s.model.host, s.model.api, s.model.num_ctx), ("http://localhost:8000", "openai", 32768))
        self.assertEqual(s.model.temperature, 0.5)                  # untouched by the preset
        self.assertEqual(src["model.host"], "preset cluster")
        self.assertEqual(s.preset, "cluster")
        with self.assertRaises(ConfigError):
            load_settings(self.path, preset="nope", env={})

    def test_environment_then_flags_win(self) -> None:
        env = {"STRANGER_MODEL_TEMPERATURE": "0.9", "STRANGER_VOICE_ENABLED": "yes", "STRANGER_WEB_PORT": "9000",
               "STRANGER_MODEL_MAX_TOKENS": "50", "STRANGER_SPEECH_DEVICE": "Digital Microphone",
               "STRANGER_PIPER_VOICES": "/ignored", "STRANGER_CONFIG": str(self.path)}
        s, src = load_settings(None, env=env, flags={"web": {"port": 9100, "bind": None}, "model": {"think": False}})
        self.assertEqual(s.source_file, self.path.resolve())        # found through STRANGER_CONFIG
        self.assertEqual((s.model.temperature, s.voice.enabled, s.model.max_tokens), (0.9, True, 50))
        self.assertEqual(s.speech.device, "Digital Microphone")
        self.assertEqual((s.web.port, src["web.port"]), (9100, "flag"))
        self.assertEqual((s.web.bind, src["web.bind"]), ("127.0.0.1", "default"))
        self.assertEqual((s.model.think, src["model.think"]), (False, "flag"))
        self.assertEqual(src["model.temperature"], "environment")

    def test_errors_name_the_key(self) -> None:
        bad = Path(self.tmp.name) / "bad.toml"
        bad.write_text("[model]\nmodle = 'x'\n")
        with self.assertRaises(ConfigError) as cm:
            load_settings(bad, env={})
        self.assertIn("'modle'", str(cm.exception))
        bad.write_text("[modle]\nx = 1\n")
        with self.assertRaises(ConfigError):
            load_settings(bad, env={})
        bad.write_text("[agent]\nreply_mode = 'sometimes'\n")
        with self.assertRaises(ConfigError):
            load_settings(bad, env={})
        with self.assertRaises(ConfigError):
            load_settings(Path(self.tmp.name) / "missing.toml", env={})
        with self.assertRaises(ConfigError):
            load_settings(self.path, env={"STRANGER_WEB_PORT": "eighty"})

    def test_coercion(self) -> None:
        self.assertEqual(coerce("off", "bool", "x"), False)
        self.assertEqual(coerce(3, "float", "x"), 3.0)
        self.assertEqual(coerce("a, b", "tuple[str, ...]", "x"), ("a", "b"))
        self.assertEqual(coerce("12", "str | int | None", "x"), 12)
        self.assertEqual(coerce("none", "str | None", "x"), None)
        self.assertEqual(coerce(None, "bool | None", "x"), None)

    def test_table_lists_every_setting_with_its_source(self) -> None:
        s, src = load_settings(self.path, preset="quiet", env={})
        text = format_settings(s, src)
        self.assertIn("[model]", text)
        self.assertIn("preset quiet", text)
        self.assertIn("temperature", text)
        self.assertIn("(live)", text)
        self.assertIn(str(self.path.resolve()), text)

    def test_repo_default_file_loads(self) -> None:
        repo_file = Path(__file__).resolve().parent.parent / "stranger.toml"
        s, src = load_settings(repo_file, env={})
        self.assertEqual(s.agent.user_name, "Visitor")
        self.assertEqual(s.model.model, "qwen3:1.7b")
        for preset in ("cluster", "fast", "typed"):
            load_settings(repo_file, preset=preset, env={})
