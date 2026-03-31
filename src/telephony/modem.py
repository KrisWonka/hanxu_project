"""
4G Modem AT command interface (SIM7600CE).

Modes:
  mock:   prints commands (Mac dev)
  real:   sends AT commands via pyserial (Orange Pi)
  aliyun: uses Alibaba Cloud voice API (requires enterprise credentials)
"""

from __future__ import annotations

import logging
import re
import threading
import time
from enum import Enum

logger = logging.getLogger(__name__)


class CallState(Enum):
    IDLE = "idle"
    DIALING = "dialing"
    RINGING = "ringing"
    ACTIVE = "active"
    INCOMING = "incoming"


class Modem:
    """SIM7600CE 4G modem interface via AT commands."""

    def __init__(
        self,
        mode: str = "mock",
        port: str = "/dev/ttyUSB2",
        baud_rate: int = 115200,
        timeout: int = 1,
        on_incoming_call=None,
    ):
        self.mode = mode
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self._serial = None
        self._call_state = CallState.IDLE
        self._incoming_number: str | None = None
        self._on_incoming_call = on_incoming_call
        self._monitor_thread: threading.Thread | None = None
        self._running = False

        self._aliyun: object | None = None

        if mode == "real":
            self._open_serial()
            self._enable_caller_id()
            self._start_monitor()
        elif mode == "aliyun":
            self._init_aliyun()

    def _init_aliyun(self):
        try:
            from src.telephony.aliyun_voice import AliyunVoice
            self._aliyun = AliyunVoice()
            logger.info("Alibaba Cloud Voice API initialized")
        except Exception as e:
            logger.error("Failed to init Aliyun voice: %s — falling back to mock", e)
            self.mode = "mock"

    def _open_serial(self):
        try:
            import serial  # type: ignore

            self._serial = serial.Serial(
                self.port, self.baud_rate, timeout=self.timeout
            )
            logger.info("Modem serial opened: %s @ %d", self.port, self.baud_rate)
        except Exception as e:
            logger.error("Failed to open serial %s: %s", self.port, e)
            self.mode = "mock"

    def _enable_caller_id(self):
        """Enable CLIP (Calling Line Identification Presentation) for incoming call number display."""
        self._send_at("AT+CLIP=1", wait=0.5)

    def _start_monitor(self):
        """Start a background thread to monitor for unsolicited responses (incoming calls, etc.)."""
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Modem monitor thread started")

    def _monitor_loop(self):
        """Read serial for unsolicited result codes like RING, +CLIP, NO CARRIER."""
        while self._running and self._serial:
            try:
                if self._serial.in_waiting > 0:
                    line = self._serial.readline().decode(errors="ignore").strip()
                    if line:
                        self._handle_urc(line)
                else:
                    time.sleep(0.2)
            except Exception as e:
                logger.error("Monitor error: %s", e)
                time.sleep(1)

    def _handle_urc(self, line: str):
        """Handle Unsolicited Result Codes."""
        logger.debug("URC: %s", line)

        if "RING" in line:
            self._call_state = CallState.INCOMING
            logger.info("Incoming call detected")

        clip_match = re.search(r'\+CLIP:\s*"(\+?\d+)"', line)
        if clip_match:
            self._incoming_number = clip_match.group(1)
            logger.info("Caller ID: %s", self._incoming_number)
            if self._on_incoming_call:
                self._on_incoming_call(self._incoming_number)

        if "NO CARRIER" in line:
            self._call_state = CallState.IDLE
            self._incoming_number = None
            logger.info("Call ended (NO CARRIER)")

        if "BUSY" in line:
            self._call_state = CallState.IDLE
            logger.info("Remote busy")

    def _send_at(self, command: str, wait: float = 1.0) -> str:
        """Send an AT command and return the response."""
        if self.mode == "mock":
            logger.info("[MOCK AT] >>> %s", command)
            print(f"  📡 [模拟AT指令] {command}")
            return "OK"

        if not self._serial:
            raise RuntimeError("Serial port not available")

        self._serial.write((command + "\r\n").encode())
        time.sleep(wait)

        response = self._serial.read(self._serial.in_waiting).decode(errors="ignore")
        logger.info("AT >>> %s | <<< %s", command, response.strip())
        return response

    @property
    def call_state(self) -> CallState:
        return self._call_state

    @property
    def incoming_number(self) -> str | None:
        return self._incoming_number

    def check_signal(self) -> dict:
        """Query signal quality. Returns parsed result."""
        response = self._send_at("AT+CSQ")
        if self.mode == "mock":
            return {"success": True, "rssi": 20, "quality": "good"}
        match = re.search(r"\+CSQ:\s*(\d+),(\d+)", response)
        if match:
            rssi = int(match.group(1))
            quality = "excellent" if rssi > 20 else "good" if rssi > 10 else "poor" if rssi > 5 else "none"
            return {"success": True, "rssi": rssi, "quality": quality}
        return {"success": False, "raw": response}

    def check_registration(self) -> dict:
        """Check network registration status."""
        response = self._send_at("AT+CREG?")
        if self.mode == "mock":
            return {"success": True, "registered": True, "roaming": False}
        match = re.search(r"\+CREG:\s*\d+,(\d+)", response)
        if match:
            stat = int(match.group(1))
            return {
                "success": True,
                "registered": stat in (1, 5),
                "roaming": stat == 5,
            }
        return {"success": False, "raw": response}

    def get_call_status(self) -> dict:
        """Query current call status via AT+CLCC."""
        if self.mode == "mock":
            return {"success": True, "state": self._call_state.value}

        response = self._send_at("AT+CLCC")
        calls = re.findall(r"\+CLCC:\s*(\d+),(\d+),(\d+),(\d+),(\d+)", response)
        if calls:
            states = {0: "active", 1: "held", 2: "dialing", 3: "alerting", 4: "incoming", 5: "waiting"}
            call = calls[0]
            stat = int(call[2])
            self._call_state = CallState.ACTIVE if stat == 0 else CallState.DIALING if stat in (2, 3) else CallState.INCOMING if stat == 4 else CallState.IDLE
            return {
                "success": True,
                "state": states.get(stat, "unknown"),
                "direction": "outgoing" if call[1] == "0" else "incoming",
            }

        self._call_state = CallState.IDLE
        return {"success": True, "state": "idle"}

    def answer_call(self) -> bool:
        """Answer an incoming call."""
        logger.info("Answering call")
        response = self._send_at("ATA", wait=2.0)
        if self.mode == "mock":
            self._call_state = CallState.ACTIVE
            print("  📞 [模拟接听] 已接听")
            return True
        if "OK" in response or "CONNECT" in response:
            self._call_state = CallState.ACTIVE
            return True
        return False

    def make_call(self, phone_number: str) -> bool:
        phone_number = phone_number.strip().replace(" ", "").replace("-", "")

        if not phone_number.replace("+", "").isdigit() or len(phone_number) < 5:
            logger.error("Invalid phone number: %s", phone_number)
            return False

        if self.mode == "aliyun" and self._aliyun:
            result = self._aliyun.click_to_dial(caller="", callee=phone_number)
            return result.get("success", False)

        logger.info("Dialing %s ...", phone_number)
        self._call_state = CallState.DIALING
        response = self._send_at(f"ATD{phone_number};", wait=2.0)

        if self.mode == "mock":
            print(f"  📞 [模拟拨号] 正在呼叫 {phone_number} ...")
            return True

        if "OK" in response or "CONNECT" in response:
            return True
        self._call_state = CallState.IDLE
        return False

    def hang_up(self) -> bool:
        logger.info("Hanging up")
        response = self._send_at("ATH")
        self._call_state = CallState.IDLE
        self._incoming_number = None
        if self.mode == "mock":
            print("  📴 [模拟挂断] 已挂断")
            return True
        return "OK" in response

    def send_sms(self, phone_number: str, message: str) -> bool:
        phone_number = phone_number.strip().replace(" ", "").replace("-", "")
        logger.info("Sending SMS to %s: %s", phone_number, message)

        self._send_at("AT+CMGF=1", wait=0.5)
        self._send_at(f'AT+CMGS="{phone_number}"', wait=1.0)

        if self.mode == "mock":
            print(f"  ✉️  [模拟短信] → {phone_number}: {message}")
            return True

        if not self._serial:
            return False

        self._serial.write((message + "\x1a").encode())
        time.sleep(3)
        response = self._serial.read(self._serial.in_waiting).decode(errors="ignore")
        return "OK" in response

    def close(self):
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        if self._serial:
            self._serial.close()
            logger.info("Modem serial closed")
