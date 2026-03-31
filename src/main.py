"""
Voice Agent — main entry point.

Modes:
  text: pure text I/O (best for Docker / no audio hardware)
  dev:  Mac with mic (keyboard trigger + voice recording + TTS playback)
  prod: Orange Pi 5 (GPIO button + mic + speaker + real 4G modem)
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.agent.core import Agent
from src.telephony.contacts import ContactBook
from src.telephony.modem import Modem

logger = logging.getLogger(__name__)


def load_config() -> dict:
    profile = os.environ.get("VOICE_AGENT_PROFILE", "")
    if profile:
        config_path = ROOT_DIR / "config" / f"settings.{profile}.yaml"
    else:
        config_path = ROOT_DIR / "config" / "settings.yaml"

    if not config_path.exists():
        config_path = ROOT_DIR / "config" / "settings.yaml"

    logger.info("Loading config: %s", config_path.name)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_text_mode(agent: Agent):
    """Pure text mode — type commands, see responses. No audio needed."""
    print("(输入 'quit' 或 Ctrl+C 退出)\n")

    while True:
        try:
            text = input("你: ").strip()
        except EOFError:
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            break

        print("🤖 思考中...")
        result = agent.run(text)

        print(f"💬 助手: {result.reply}")
        if result.tool_calls_made:
            print(f"   🔧 执行了: {', '.join(result.tool_calls_made)}")
        print()


def run_voice_mode(agent: Agent, config: dict, mode: str):
    """Voice mode — record audio, STT, Agent, TTS."""
    from src.audio.recorder import Recorder
    from src.audio.tts import TTS
    from src.hardware.button import Button, LED
    from src.stt.engine import STTEngine

    button = Button(
        mode=mode,
        chip=config["gpio"]["chip"],
        pin=config["gpio"]["button_pin"],
    )

    led = LED(
        mode=mode,
        chip=config["gpio"]["chip"],
        pin=config["gpio"].get("led_pin", 11),
    )

    recorder = Recorder(
        sample_rate=config["audio"]["sample_rate"],
        channels=config["audio"]["channels"],
        silence_timeout=config["audio"]["silence_timeout"],
        silence_threshold=config["audio"]["silence_threshold"],
        input_device=config["audio"].get("input_device"),
    )

    stt = STTEngine(
        provider=config["stt"]["provider"],
        language=config["stt"]["language"],
        whisper_model=config["stt"].get("whisper_model", "medium"),
    )

    tts = TTS(
        provider=config["tts"]["provider"],
        voice=config["tts"]["voice"],
        rate=config["tts"]["rate"],
    )

    try:
        while True:
            led.off()
            button.wait_for_press()
            led.on()

            if mode == "dev":
                print("🎤 正在录音... (按 Enter 停止，或等待静音自动停止)")
                audio_bytes = recorder.record_with_silence_detection()
            else:
                audio_bytes = b""

                def _record():
                    nonlocal audio_bytes
                    audio_bytes = recorder.record_with_silence_detection()

                rec_thread = threading.Thread(target=_record, daemon=True)
                rec_thread.start()

                button.wait_for_release()
                recorder.stop()
                rec_thread.join(timeout=2)

            if not audio_bytes:
                led.off()
                print("⚠️  未录到声音，请重试")
                continue

            led.blink(0.3)
            print("🔍 识别中...")
            text = stt.recognize(audio_bytes)

            if not text:
                led.off()
                print("⚠️  未识别到有效语音，请重试")
                continue

            print(f"📝 你说: {text}")

            print("🤖 思考中...")
            result = agent.run(text)

            led.on()
            print(f"💬 助手: {result.reply}")
            if result.tool_calls_made:
                print(f"   🔧 执行了: {', '.join(result.tool_calls_made)}")

            tts.speak(result.reply)

    finally:
        led.close()
        button.close()


def main():
    setup_logging()
    config = load_config()
    mode = config.get("mode", "text")
    logger.info("Starting Voice Agent in [%s] mode", mode)

    contacts = ContactBook(ROOT_DIR / "config" / "contacts.yaml")

    tel_cfg = config["telephony"]
    modem = Modem(
        mode=tel_cfg["modem_mode"],
        port=tel_cfg["serial_port"],
        baud_rate=tel_cfg["baud_rate"],
        timeout=tel_cfg["timeout"],
    )

    agent = Agent(
        provider=config["agent"]["provider"],
        model=config["agent"]["model"],
        temperature=config["agent"]["temperature"],
        max_tokens=config["agent"]["max_tokens"],
        contact_book=contacts,
        modem=modem,
    )

    print("=" * 50)
    print("  语音助手已启动")
    print(f"  模式: {mode}")
    print(f"  LLM:  {config['agent']['provider']} / {config['agent']['model']}")
    if mode != "text":
        print(f"  STT:  {config['stt']['provider']}")
        print(f"  TTS:  {config['tts']['provider']}")
    print("=" * 50)

    try:
        if mode == "text":
            run_text_mode(agent)
        else:
            run_voice_mode(agent, config, mode)
    except KeyboardInterrupt:
        print("\n👋 再见！")
    finally:
        modem.close()


if __name__ == "__main__":
    main()
