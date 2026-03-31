"""
AI Agent core — LLM with function calling.

Supports OpenAI-compatible APIs (DeepSeek, DashScope/通义千问, OpenAI, Ollama).
Executes tool calls (make_phone_call, hang_up, send_sms, list_contacts)
and returns the final text reply for TTS.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from openai import OpenAI

from src.agent.prompts import SYSTEM_PROMPT, TOOLS
from src.telephony.contacts import ContactBook
from src.telephony.modem import Modem

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    reply: str
    tool_calls_made: list[str]


class Agent:
    """Voice agent powered by LLM function calling."""

    def __init__(
        self,
        provider: str = "dashscope",
        model: str = "qwen-plus",
        temperature: float = 0.3,
        max_tokens: int = 512,
        contact_book: ContactBook | None = None,
        modem: Modem | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.contact_book = contact_book or ContactBook()
        self.modem = modem or Modem(mode="mock")

        providers = {
            "deepseek": {
                "api_key_env": "DEEPSEEK_API_KEY",
                "base_url": "https://api.deepseek.com",
            },
            "dashscope": {
                "api_key_env": "DASHSCOPE_API_KEY",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            },
            "ollama": {
                "api_key_env": None,
                "base_url": "http://127.0.0.1:11434/v1",
            },
            "openai": {
                "api_key_env": "OPENAI_API_KEY",
                "base_url": None,
            },
        }

        cfg = providers.get(provider, providers["openai"])

        api_key = (
            os.environ.get(cfg["api_key_env"], "sk-placeholder")
            if cfg["api_key_env"]
            else "ollama"
        )
        kwargs: dict = {"api_key": api_key}
        if cfg["base_url"]:
            kwargs["base_url"] = cfg["base_url"]

        self.client = OpenAI(**kwargs)

        self._history: list[dict] = [
            {"role": "system", "content": self._build_system_prompt()},
        ]

    def _build_system_prompt(self) -> str:
        contacts_info = self.contact_book.list_all()
        contacts_str = "\n".join(
            f"- {c['name']}（{', '.join(c['aliases'])}）: {c['phone']}"
            if c["aliases"]
            else f"- {c['name']}: {c['phone']}"
            for c in contacts_info
        )
        return (
            SYSTEM_PROMPT
            + f"\n\n当前通讯录:\n{contacts_str}\n"
        )

    def run(self, user_text: str) -> AgentResult:
        """
        Process user input through the LLM, handle any tool calls,
        and return the final assistant reply.
        """
        logger.info("User: %s", user_text)
        self._history.append({"role": "user", "content": user_text})

        tool_calls_made: list[str] = []
        max_rounds = 5  # safety limit for tool-call loops

        for _ in range(max_rounds):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._history,
                tools=TOOLS,
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            message = response.choices[0].message

            if not message.tool_calls:
                reply = message.content or ""
                self._history.append({"role": "assistant", "content": reply})
                logger.info("Agent: %s", reply)
                return AgentResult(reply=reply, tool_calls_made=tool_calls_made)

            self._history.append(message.model_dump())

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                logger.info("Tool call: %s(%s)", fn_name, fn_args)

                result = self._execute_tool(fn_name, fn_args)
                tool_calls_made.append(fn_name)

                self._history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        reply = "抱歉，处理超时了，请再说一次。"
        self._history.append({"role": "assistant", "content": reply})
        return AgentResult(reply=reply, tool_calls_made=tool_calls_made)

    def _execute_tool(self, name: str, args: dict) -> dict:
        """Dispatch and execute a tool call, return result dict."""

        if name == "make_phone_call":
            return self._tool_make_call(args["name_or_number"])
        elif name == "hang_up":
            return self._tool_hang_up()
        elif name == "send_sms":
            return self._tool_send_sms(args["name_or_number"], args["message"])
        elif name == "list_contacts":
            return self._tool_list_contacts()
        else:
            return {"success": False, "error": f"Unknown tool: {name}"}

    def _tool_make_call(self, name_or_number: str) -> dict:
        phone = name_or_number
        contact_name = None

        if not name_or_number.replace("+", "").isdigit():
            contact = self.contact_book.lookup(name_or_number)
            if not contact:
                matches = self.contact_book.fuzzy_lookup(name_or_number)
                if len(matches) == 1:
                    contact = matches[0]
                elif len(matches) > 1:
                    names = ", ".join(m.name for m in matches)
                    return {
                        "success": False,
                        "error": f"找到多个匹配: {names}，请说具体一点",
                    }
                else:
                    return {
                        "success": False,
                        "error": f"通讯录中没有找到「{name_or_number}」",
                    }
            phone = contact.phone
            contact_name = contact.name

        success = self.modem.make_call(phone)
        return {
            "success": success,
            "phone": phone,
            "contact_name": contact_name,
        }

    def _tool_hang_up(self) -> dict:
        success = self.modem.hang_up()
        return {"success": success}

    def _tool_send_sms(self, name_or_number: str, message: str) -> dict:
        phone = name_or_number

        if not name_or_number.replace("+", "").isdigit():
            contact = self.contact_book.lookup(name_or_number)
            if not contact:
                return {
                    "success": False,
                    "error": f"通讯录中没有找到「{name_or_number}」",
                }
            phone = contact.phone

        success = self.modem.send_sms(phone, message)
        return {"success": success, "phone": phone}

    def _tool_list_contacts(self) -> dict:
        return {"contacts": self.contact_book.list_all()}

    def reset_history(self):
        """Clear conversation history (keep system prompt)."""
        self._history = [self._history[0]]
