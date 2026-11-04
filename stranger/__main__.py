"""Command line: `python -m stranger <command>`.

    serve      web GUI (default port 8766), microphone in unless --input text
    chat       the same agent in the terminal
    check      is the model endpoint reachable, which models and devices are there
    profiles   list the profiles found
"""

from __future__ import annotations

import argparse
import socket
import sys

from . import __version__
from .agent import Agent, AgentSettings
from .llm import LLMClient, LLMConfig, check_endpoint, installed_models
from .profiles import ProfileError, build_system_prompt, list_profiles, load_profile


def add_agent_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--model", default=LLMConfig.model, help="model name on the endpoint")
    g.add_argument("--host", default=LLMConfig.host, help="Ollama or OpenAI-compatible server URL")
    g.add_argument("--api", choices=["ollama", "openai"], default="ollama", help="wire protocol of the endpoint")
    g.add_argument("--num-ctx", type=int, default=LLMConfig.num_ctx, help="context window to request / meter against")
    g.add_argument("--max-tokens", type=int, default=LLMConfig.max_tokens)
    g.add_argument("--max-sentences", type=int, default=LLMConfig.max_sentences, help="0 for no limit")
    g.add_argument("--no-thinking", action="store_true", help="don't ask the model for a thinking stream")
    g.add_argument("--no-warm-up", action="store_true", help="skip pre-evaluating the system prompt at start")
    a = p.add_argument_group("agent")
    a.add_argument("--profile", default="stranger", help="profile name under profiles/, or a directory path")
    a.add_argument("--name", default="Visitor", help="your name, as the model sees it")
    a.add_argument("--trigger-name", default=None, help="word that makes the agent answer (default: the display name's last word)")
    a.add_argument("--reply", choices=["named", "always"], default="named", help="answer only when named, or always")
    s = p.add_argument_group("speech in")
    s.add_argument("--input", choices=["mic", "text"], default="mic")
    s.add_argument("--mic-device", default=None, help="input device index or name substring")
    s.add_argument("--whisper-model", default="small")
    s.add_argument("--chunk-seconds", type=float, default=2.0)
    s.add_argument("--max-chunks", type=int, default=6, help="close an utterance after this many speech chunks")
    v = p.add_argument_group("speech out")
    v.add_argument("--speak", action="store_true", help="read replies aloud (thinking is never spoken)")
    v.add_argument("--tts", default="piper", help="piper (offline) | espeak | elevenlabs | openai")
    v.add_argument("--voice", default=None, help="voice name or path, e.g. en_GB-alan-medium")
    v.add_argument("--output-device", default=None, help="speaker device index or name substring")


def build_agent(args) -> tuple[Agent, LLMClient]:
    profile = load_profile(args.profile)
    config = LLMConfig(model=args.model, host=args.host, api=args.api, num_ctx=args.num_ctx, max_tokens=args.max_tokens,
                       max_sentences=args.max_sentences, think=not args.no_thinking,
                       stop=(f"{args.name}:", f"{args.name.upper()}:") + tuple(profile.stop_words))
    client = LLMClient(config)
    speak = stop_speech = None
    if args.speak:
        from .voice import AudioPlayer, VoiceConfig, make_tts
        tts = make_tts(VoiceConfig(backend=args.tts, voice=args.voice, output_device=args.output_device))
        player = AudioPlayer(args.output_device)
        print(f"Speech: {tts.name} -> {player.device_name}")

        def speak(text: str) -> None:
            player.play(tts.synthesize(text), block=True)
        stop_speech = player.stop
    trigger = args.trigger_name or profile.display_name.split()[-1]
    settings = AgentSettings(user_name=args.name, display_name=profile.display_name, speaker_tag=profile.speaker_tag,
                             trigger_name=trigger, reply_mode=args.reply)
    agent = Agent(client, build_system_prompt(profile, args.name), settings, speak=speak, stop_speech=stop_speech)
    print(f"Profile {profile}; answers when '{trigger}' is said" if args.reply == "named" else f"Profile {profile}; answers always")
    return agent, client


def warm_up(agent: Agent, client: LLMClient, skip: bool) -> None:
    online, detail = client.check()
    print(f"Model endpoint {client.config.host} ({client.config.api}): {'online, ' + detail.splitlines()[0] if online else 'OFFLINE'}")
    if not online:
        print(detail)
        return
    if not skip:
        print(f"Warming up {client.config.model} (pre-evaluating the system prompt)...")
        try:
            print(f"  ready in {client.warm_up(agent.system_prompt):.1f}s")
        except Exception as e:
            print(f"  warm-up failed: {e}")


def speech_config(args):
    from .speech import SpeechConfig
    return SpeechConfig(model_size=args.whisper_model, device=args.mic_device, chunk_seconds=args.chunk_seconds)


def cmd_serve(args) -> int:
    agent, client = build_agent(args)
    warm_up(agent, client, args.no_warm_up)
    with socket.socket() as probe:
        if probe.connect_ex((args.bind, args.port)) == 0:
            print(f"Port {args.port} on {args.bind} is already in use by another program (left running); "
                  f"try --port {args.port + 1}")
            return 2
    from .chat import TerminalView
    from .web.server import create_app, mic_feed_factory
    if not args.quiet:
        TerminalView(agent).start()
    mic_feed = mic_feed_factory(speech_config(args), args.max_chunks) if args.input == "mic" else None
    app = create_app(agent, endpoint_check=client.check, mic_feed=mic_feed)
    print(f"Open http://{args.bind}:{args.port}")
    import uvicorn
    uvicorn.run(app, host=args.bind, port=args.port, log_level="warning")
    return 0


def cmd_chat(args) -> int:
    from .chat import TerminalView, run_mic, run_text
    agent, client = build_agent(args)
    warm_up(agent, client, args.no_warm_up)
    view = TerminalView(agent, show_thinking=not args.no_thinking)
    try:
        if args.input == "text":
            run_text(agent, view)
        else:
            run_mic(agent, view, speech_config(args), args.max_chunks)
    except KeyboardInterrupt:
        print("\n[bye]")
    return 0


def cmd_check(args) -> int:
    online, detail = check_endpoint(args.host, args.api)
    print(f"Model endpoint {args.host} ({args.api}): {'online, ' + detail if online else 'OFFLINE'}")
    if not online:
        print(detail)
    else:
        try:
            print("  models: " + ", ".join(installed_models(args.host, args.api)))
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
    serve.add_argument("--bind", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--quiet", action="store_true", help="don't echo the conversation to this terminal")
    chat = sub.add_parser("chat", help="terminal chat")
    add_agent_args(chat)
    check = sub.add_parser("check", help="endpoint, models, devices, voices")
    check.add_argument("--host", default=LLMConfig.host)
    check.add_argument("--api", choices=["ollama", "openai"], default="ollama")
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
        if args.command == "check":
            return cmd_check(args)
        for name in list_profiles():
            print(f"  {load_profile(name)}")
        return 0
    except (ProfileError, RuntimeError) as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
