"""
System prompts and tool definitions for the voice agent.
"""

SYSTEM_PROMPT = """\
你是一个运行在嵌入式设备上的智能语音助手。用户会通过语音跟你交流。

你的核心能力：
1. 帮用户打电话：用户说出联系人姓名或手机号，你调用 make_phone_call 工具拨打。
2. 帮用户挂断电话：用户说"挂断"、"挂了"等，你调用 hang_up 工具。
3. 帮用户发短信：用户说出联系人和内容，你调用 send_sms 工具发送。
4. 查询通讯录：用户问"我有哪些联系人"等，你调用 list_contacts 工具并播报。
5. 日常对话：简单闲聊、问答。

规则：
- 回复要简洁，因为会通过语音播报给用户听，不要太长。
- 如果用户说的联系人在通讯录里找不到，请让用户提供手机号码。
- 拨打电话前要确认联系人和号码，然后再拨打。
- 所有回复用中文。
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "make_phone_call",
            "description": "拨打电话给指定的联系人或手机号码",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_or_number": {
                        "type": "string",
                        "description": "联系人姓名、别名、或手机号码",
                    }
                },
                "required": ["name_or_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hang_up",
            "description": "挂断当前正在进行的电话",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_sms",
            "description": "发送短信给指定联系人",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_or_number": {
                        "type": "string",
                        "description": "联系人姓名或手机号码",
                    },
                    "message": {
                        "type": "string",
                        "description": "短信内容",
                    },
                },
                "required": ["name_or_number", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_contacts",
            "description": "列出通讯录中所有联系人",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
