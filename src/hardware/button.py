"""
Physical button + LED input (GPIO on Orange Pi, keyboard fallback on Mac).

In dev mode:  press Enter to simulate button press/release, LED is printed.
In prod mode: uses gpiod to listen for GPIO edge events and drive an LED.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class LED:
    """Status LED controller."""

    def __init__(self, mode: str = "dev", chip: str = "gpiochip0", pin: int = 11):
        self._mode = mode
        self._chip = None
        self._line = None
        self._blinking = False

        if mode == "prod":
            self._setup_gpio(chip, pin)

    def _setup_gpio(self, chip: str, pin: int):
        try:
            import gpiod  # type: ignore

            self._chip = gpiod.Chip(chip)
            self._line = self._chip.get_line(pin)
            self._line.request(consumer="voice-agent-led", type=gpiod.LINE_REQ_DIR_OUT)
            logger.info("LED ready on %s pin %d", chip, pin)
        except Exception as e:
            logger.warning("LED GPIO init failed: %s", e)
            self._mode = "dev"

    def on(self):
        if self._mode == "dev":
            return
        if self._line:
            self._line.set_value(1)

    def off(self):
        self._blinking = False
        if self._mode == "dev":
            return
        if self._line:
            self._line.set_value(0)

    def blink(self, interval: float = 0.5):
        """Blink LED in background."""
        self._blinking = True

        def _blink_loop():
            import time
            while self._blinking:
                self.on()
                time.sleep(interval)
                if not self._blinking:
                    break
                if self._line:
                    self._line.set_value(0)
                time.sleep(interval)

        t = threading.Thread(target=_blink_loop, daemon=True)
        t.start()

    def close(self):
        self._blinking = False
        if self._line:
            self._line.set_value(0)
            self._line.release()
        if self._chip:
            self._chip.close()


class Button:
    """Unified button interface with dev/prod backends."""

    def __init__(self, mode: str = "dev", chip: str = "gpiochip0", pin: int = 7):
        self._mode = mode
        self._chip_name = chip
        self._pin = pin
        self._chip = None
        self._line = None

        if mode == "prod":
            self._setup_gpio()

    def _setup_gpio(self):
        try:
            import gpiod  # type: ignore

            self._chip = gpiod.Chip(self._chip_name)
            self._line = self._chip.get_line(self._pin)
            self._line.request(
                consumer="voice-agent",
                type=gpiod.LINE_REQ_EV_BOTH_EDGES,
            )
            logger.info("GPIO button ready on %s pin %d", self._chip_name, self._pin)
        except Exception as e:
            logger.error("Failed to init GPIO: %s — falling back to keyboard", e)
            self._mode = "dev"

    def wait_for_press(self) -> None:
        """Block until the button is pressed (or Enter is hit in dev mode)."""
        if self._mode == "dev":
            input("\n🎙  按 Enter 开始说话...")
        else:
            self._wait_gpio_edge(target_value=1)

    def wait_for_release(self) -> None:
        """Block until the button is released (or Enter is hit in dev mode)."""
        if self._mode == "dev":
            input("🛑  说完后按 Enter 停止录音...")
        else:
            self._wait_gpio_edge(target_value=0)

    def _wait_gpio_edge(self, target_value: int):
        import gpiod  # type: ignore

        while True:
            if self._line.event_wait(sec=1):
                event = self._line.event_read()
                if target_value == 1 and event.type == gpiod.LineEvent.RISING_EDGE:
                    return
                if target_value == 0 and event.type == gpiod.LineEvent.FALLING_EDGE:
                    return

    def close(self):
        if self._line:
            self._line.release()
        if self._chip:
            self._chip.close()
