"""
4G Modem AT command interface.

In mock mode (Mac dev): prints commands instead of sending to serial port.
In real mode (Orange Pi): sends AT commands to SIM7600CE via pyserial.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class Modem:
    """SIM7600CE 4G modem interface via AT commands."""

    def __init__(
        self,
        mode: str = "mock",
        port: str = "/dev/ttyUSB2",
        baud_rate: int = 115200,
        timeout: int = 1,
    ):
        self.mode = mode
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self._serial = None

        if mode == "real":
            self._open_serial()

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

    def check_signal(self) -> str:
        """Query signal quality. Returns raw AT response."""
        return self._send_at("AT+CSQ")

    def check_registration(self) -> str:
        """Check network registration status."""
        return self._send_at("AT+CREG?")

    def make_call(self, phone_number: str) -> bool:
        """
        Initiate a voice call.
        Returns True if the call was initiated successfully.
        """
        phone_number = phone_number.strip().replace(" ", "").replace("-", "")

        if not phone_number.isdigit() or len(phone_number) < 5:
            logger.error("Invalid phone number: %s", phone_number)
            return False

        logger.info("Dialing %s ...", phone_number)
        response = self._send_at(f"ATD{phone_number};", wait=2.0)

        if self.mode == "mock":
            print(f"  📞 [模拟拨号] 正在呼叫 {phone_number} ...")
            return True

        return "OK" in response or "CONNECT" in response

    def hang_up(self) -> bool:
        """Hang up the current call."""
        logger.info("Hanging up")
        response = self._send_at("ATH")
        if self.mode == "mock":
            print("  📴 [模拟挂断] 已挂断")
            return True
        return "OK" in response

    def send_sms(self, phone_number: str, message: str) -> bool:
        """Send an SMS message (for future expansion)."""
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
        if self._serial:
            self._serial.close()
            logger.info("Modem serial closed")
