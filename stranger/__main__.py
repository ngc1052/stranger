"""Command line: `python -m stranger <command>`.

    serve      web GUI (default port 8766)
    chat       the same agent in the terminal
    config     the effective settings and where each value came from
    check      is the model endpoint reachable, which models and devices are there
    profiles   list the profiles found

Settings come from stranger.toml, a preset, STRANGER_* environment variables and
flags, in that order (see docs/configuration.md). Flags given here override the file.
"""

from __future__ import annotations

import argparse
import socket
import sys

from . import __version__
from .agent import Agent, AgentSettings
from .config import ConfigError, LIVE_KEYS, Settings, format_settings, load_settings
from .llm import LLMClient, check_endpoint, installed_models
from .profiles import ProfileError, build_system_prompt, list_profiles, load_profile

# flag -> (section, key); a flag left unset does not override the file
FLAGS = {
    "model": ("model", "model"), "host": ("model", "host"), "api": ("model", "api"), "num_ctx": ("model", "num_ctx"),
    "max_tokens": ("model", "max_tokens"), "max_sentences": ("model", "max_sentences"),
    "temperature": ("model", "temperature"),
    "profile": ("agent", "profile"), "name": ("agent", "user_name"), "trigger_name": ("agent", "trigger_name"),
    "reply": ("agent", "reply_mode"),
    "input": ("speech", "input"), "mic_device": ("speech", "device"), "whisper_model": ("speech", "model_size"),
    "chunk_seconds": ("speech", "chunk_seconds"), "max_chunks": ("speech", "max_chunks"),
    "tts": ("voice", "backend"), "voice": ("voice", "voice"), "output_device": ("voice", "output_device"),
    "bind": ("web", "bind"), "port": ("web", "port"),
}


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="settings file (default: ./stranger.toml or ~/.config/stranger/stranger.toml)")
    p.add_argument("--preset", default=None, help="a [presets.NAME] section of the settings file to overlay")


def add_agent_args(p: argparse.ArgumentParser) -> None:
    add_common_args(p)
    g = p.add_argument_group("model (override [model])")
    g.add_argument("--model"); g.add_argument("--host"); g.add_argument("--api", choices=["ollama", "openai"])
    g.add_argument("--num-ctx", type=int); g.add_argument("--max-tokens", type=int)
    g.add_argument("--max-sentences", type=int, help="0 for no limit"); g.add_argument("--temperature", type=float)
    g.add_argument("--no-thinking", action="store_true", help="model.think = false")
    g.add_argument("--no-warm-up", action="store_true", help="agent.warm_up = false")
    a = p.add_argument_group("agent (override [agent])")
    a.add_argument("--profile"); a.add_argument("--name", help="the human's name, as the model sees it")
    a.add_argument("--trigger-name"); a.add_argument("--reply", choices=["named", "always"])
    s = p.add_argument_group("speech in (override [speech])")
    s.add_argument("--input", choices=["mic", "text"]); s.add_argument("--mic-device")
    s.add_argument("--whisper-model"); s.add_argument("--chunk-seconds", type=float); s.add_argument("--max-chunks", type=int)
    v = p.add_argument_group("speech out (override [voice])")
    v.add_argument("--speak", action="store_true", help="voice.enabled = true")
    v.add_argument("--no-speak", action="store_true", help="voice.enabled = false")
    v.add_argument("--tts"); v.add_argument("--voice"); v.add_argument("--output-device")


def flags_from(args) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for flag, (section, key) in FLAGS.items():
        value = getattr(args, flag, None)
        if value is not None:
            out.setdefault(section, {})[key] = value
    if getattr(args, "no_thinking", False):
        out.setdefault("model", {})["think"] = False
    if getattr(args, "no_warm_up", False):
        out.setdefault("agent", {})["warm_up"] = False
    if getattr(args, "speak", False):
        out.setdefault("voice", {})["enabled"] = True
    if getattr(args, "no_speak", False):
        out.setdefault("voice", {})["enabled"] = False
    if getattr(args, "quiet", False):
        out.setdefault("web", {})["quiet"] = True
    return out


def settings_from(args) -> tuple[Settings, dict]:
    return load_settings(args.config, args.preset, flags_from(args))


def build_agent(settings: Settings) -> tuple[Agent, LLMClient]:
    profile = load_profile(settings.agent.profile)
    config = settings.model
    config.stop = tuple(config.stop) + (f"{settings.agent.user_name}:", f"{settings.agent.user_name.upper()}:") \
        + tuple(profile.stop_words)
    client = LLMClient(config)
    speak = stop_speech = None
    if settings.voice.enabled:
        from .voice import AudioPlayer, make_tts
        tts = make_tts(settings.voice_config())
        player = AudioPlayer(settings.voice.output_device)
        print(f"Speech: {tts.name} -> {player.device_name}")

        def speak(text: str) -> None:
            player.play(tts.synthesize(text), block=True)
        stop_speech = player.stop
    trigger = settings.agent.trigger_name or profile.display_name.split()[-1]
    agent_settings = AgentSettings(user_name=settings.agent.user_name, display_name=profile.display_name,
                                   speaker_tag=profile.speaker_tag, trigger_name=trigger, reply_mode=settings.agent.reply_mode)
    agent = Agent(client, build_system_prompt(profile, settings.agent.user_name), agent_settings,
                  speak=speak, stop_speech=stop_speech)
    rule = f"answers when '{trigger}' is said" if settings.agent.reply_mode == "named" else "answers always"
    print(f"Settings from {settings.source_file or 'defaults'}" + (f" (preset {settings.preset})" if settings.preset else "")
          + f". Profile {profile}; {rule}.")
    return agent, client


def warm_up(agent: Agent, client: LLMClient, enabled: bool) -> None:
    online, detail = client.check()
    print(f"Model endpoint {client.config.host} ({client.config.api}): {'online, ' + detail.splitlines()[0] if online else 'OFFLINE'}")
    if not online:
        print(detail)
        return
    if enabled:
        print(f"Warming up {client.config.model} (pre-evaluating the system prompt)...")
        try:
            print(f"  ready in {client.warm_up(agent.system_prompt):.1f}s")
        except Exception as e:
            print(f"  warm-up failed: {e}")


def cmd_serve(args) -> int:
    settings, _ = settings_from(args)
    agent, client = build_agent(settings)
    warm_up(agent, client, settings.agent.warm_up)
    bind, port = settings.web.bind, settings.web.port
    with socket.socket() as probe:
        if probe.connect_ex((bind, port)) == 0:
            print(f"Port {port} on {bind} is already in use by another program (left running); try --port {port + 1}")
            return 2
    from .chat import TerminalView
    from .web.server import create_app, mic_feed_factory
    if not settings.web.quiet:
        TerminalView(agent).start()
    mic_feed = mic_feed_factory(settings.speech_config(), settings.speech.max_chunks) if settings.speech.input == "mic" else None
    app = create_app(agent, endpoint_check=client.check, mic_feed=mic_feed)
    print(f"Open http://{bind}:{port}")
    import uvicorn
    uvicorn.run(app, host=bind, port=port, log_level="warning")
    return 0


def cmd_chat(args) -> int:
    from .chat import TerminalView, run_mic, run_text
    settings, _ = settings_from(args)
    agent, client = build_agent(settings)
    warm_up(agent, client, settings.agent.warm_up)
    view = TerminalView(agent, show_thinking=bool(settings.model.think))
    try:
        if settings.speech.input == "text":
            run_text(agent, view)
        else:
            run_mic(agent, view, settings.speech_config(), settings.speech.max_chunks)
    except KeyboardInterrupt:
        print("\n[bye]")
    return 0


def cmd_config(args) -> int:
    settings, sources = settings_from(args)
    print(format_settings(settings, sources))
    print(f"\n(live) = can be changed while running, from the page or POST /api/settings: {', '.join(sorted(LIVE_KEYS))}")
    return 0


def cmd_check(args) -> int:
    settings, _ = settings_from(args)
    host, api = settings.model.host, settings.model.api
    online, detail = check_endpoint(host, api)
    print(f"Model endpoint {host} ({api}): {'online, ' + detail if online else 'OFFLINE'}")
    if not online:
        print(detail)
    else:
        try:
            print("  models: " + ", ".join(installed_models(host, api)))
        except Exception as e:
            print(f"  could not list models: {e}")
    print("Profiles: " + ", ".join(list_profiles()))
    try:
        from .speech import list_input_devices
        print("Input devices:\n" + "\n".join(f"  [{d.index}] {d.name}" for d in list_input_devices()))
    except Exception as e:
        print(f"Input devices: unavailable ({e})")
    try:
        from .voice import list_output_devices
        print("Output devices:\n" + "\n".join(f"  [{i}] {n}" for i, n in list_output_devices()))
    except Exception as e:
        print(f"Output devices: unavailable ({e})")
    try:
        from .voice.piper_tts import DEFAULT_VOICES_DIR, list_voices
        voices = list_voices(DEFAULT_VOICES_DIR)
        print(f"Piper voices in {DEFAULT_VOICES_DIR}: " + (", ".join(voices) if voices else "none (see docs/voice.md)"))
    except Exception as e:
        print(f"Piper: unavailable ({e})")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="stranger", description="The Stranger: a spoken-conversation agent on a local model.")
    p.add_argument("--version", action="version", version=f"stranger {__version__}")
    sub = p.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="web GUI")
    add_agent_args(serve)
    serve.add_argument("--bind"); serve.add_argument("--port", type=int)
    serve.add_argument("--quiet", action="store_true", help="web.quiet = true: no terminal echo")
    chat = sub.add_parser("chat", help="terminal chat")
    add_agent_args(chat)
    config = sub.add_parser("config", help="show the effective settings and their sources")
    add_agent_args(config)
    config.add_argument("--bind"); config.add_argument("--port", type=int); config.add_argument("--quiet", action="store_true")
    check = sub.add_parser("check", help="endpoint, models, devices, voices")
    add_common_args(check)
    check.add_argument("--host"); check.add_argument("--api", choices=["ollama", "openai"])
    sub.add_parser("profiles", help="list profiles")
    args = p.parse_args(argv)
    if args.command is None:
        p.print_help()
        return 0
    try:
        if args.command == "serve":
            return cmd_serve(args)
        if args.command == "chat":
            return cmd_chat(args)
        if args.command == "config":
            return cmd_config(args)
        if args.command == "check":
            return cmd_check(args)
        for name in list_profiles():
            print(f"  {load_profile(name)}")
        return 0
    except (ConfigError, ProfileError, RuntimeError) as e:
        print(f"error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
