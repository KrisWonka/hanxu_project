"""
Alibaba Cloud Voice Call API integration.

Requires:
  - Enterprise real-name verification on Alibaba Cloud
  - Purchased voice number
  - pip install alibabacloud_dyvmsapi20170525

Supports:
  - ClickToDial: Bridge call between caller and callee
  - SingleCallByTts: Call and play TTS message
  - SmartCall: AI-powered interactive call

Config in settings.yaml:
  telephony:
    modem_mode: aliyun
    aliyun:
      access_key_id: (or env ALIBABA_CLOUD_ACCESS_KEY_ID)
      access_key_secret: (or env ALIBABA_CLOUD_ACCESS_KEY_SECRET)
      called_show_number: "purchased voice number"
      tts_code: "TTS template code from console"
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class AliyunVoice:
    """Alibaba Cloud Voice Call (语音服务) wrapper."""

    def __init__(
        self,
        access_key_id: str | None = None,
        access_key_secret: str | None = None,
        called_show_number: str = "",
        tts_code: str = "",
    ):
        self.access_key_id = access_key_id or os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID", "")
        self.access_key_secret = access_key_secret or os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "")
        self.called_show_number = called_show_number
        self.tts_code = tts_code
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from alibabacloud_dyvmsapi20170525.client import Client  # type: ignore
            from alibabacloud_tea_openapi.models import Config  # type: ignore
        except ImportError:
            raise ImportError(
                "Install alibabacloud_dyvmsapi20170525: "
                "pip install alibabacloud_dyvmsapi20170525"
            )

        config = Config(
            access_key_id=self.access_key_id,
            access_key_secret=self.access_key_secret,
            endpoint="dyvmsapi.aliyuncs.com",
        )
        self._client = Client(config)
        return self._client

    def click_to_dial(self, caller: str, callee: str) -> dict:
        """
        Bridge call: platform calls both parties and connects them.
        caller: the user's phone number
        callee: the target phone number
        """
        try:
            from alibabacloud_dyvmsapi20170525.models import ClickToDialRequest  # type: ignore

            client = self._get_client()
            request = ClickToDialRequest(
                caller_show_number=self.called_show_number,
                caller_number=caller,
                called_show_number=self.called_show_number,
                called_number=callee,
            )
            response = client.click_to_dial(request)
            body = response.body
            logger.info("ClickToDial response: %s", body)
            return {
                "success": body.code == "OK",
                "call_id": body.call_id,
                "message": body.message,
            }
        except Exception as e:
            logger.error("ClickToDial failed: %s", e)
            return {"success": False, "error": str(e)}

    def single_call_by_tts(self, callee: str, tts_param: dict | None = None) -> dict:
        """
        Call the target and play a TTS message.
        tts_param: template parameters as dict, e.g. {"name": "张三", "content": "你好"}
        """
        try:
            import json
            from alibabacloud_dyvmsapi20170525.models import SingleCallByTtsRequest  # type: ignore

            client = self._get_client()
            request = SingleCallByTtsRequest(
                called_show_number=self.called_show_number,
                called_number=callee,
                tts_code=self.tts_code,
                tts_param=json.dumps(tts_param or {}, ensure_ascii=False),
            )
            response = client.single_call_by_tts(request)
            body = response.body
            logger.info("SingleCallByTts response: %s", body)
            return {
                "success": body.code == "OK",
                "call_id": body.call_id,
                "message": body.message,
            }
        except Exception as e:
            logger.error("SingleCallByTts failed: %s", e)
            return {"success": False, "error": str(e)}

    def smart_call(self, callee: str, voice_code: str = "") -> dict:
        """
        AI-powered interactive call (SmartCall).
        Requires a voice template configured in the Alibaba Cloud console.
        """
        try:
            from alibabacloud_dyvmsapi20170525.models import SmartCallRequest  # type: ignore

            client = self._get_client()
            request = SmartCallRequest(
                called_show_number=self.called_show_number,
                called_number=callee,
                voice_code=voice_code or self.tts_code,
            )
            response = client.smart_call(request)
            body = response.body
            logger.info("SmartCall response: %s", body)
            return {
                "success": body.code == "OK",
                "call_id": body.call_id,
                "message": body.message,
            }
        except Exception as e:
            logger.error("SmartCall failed: %s", e)
            return {"success": False, "error": str(e)}
