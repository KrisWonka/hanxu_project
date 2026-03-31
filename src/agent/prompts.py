"""
System prompts and tool definitions for the voice agent.
"""

SYSTEM_PROMPT = """\
你是一个语音助手，回复会被语音播报，所以必须简短（1-2句话）。

能力：打电话、挂断、发短信、管理通讯录、查天气、查时间、日常问答。

规则：
- 回复不超过30个字，除非用户要求详细说明。
- 用户说打电话，直接调用工具拨打，不要反复确认。
- 用户说的联系人找不到，简短提示并请求号码。
- 只用中文回复。
- 不要重复用户的话。
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
    {
        "type": "function",
        "function": {
            "name": "add_contact",
            "description": "添加新联系人到通讯录",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "联系人姓名",
                    },
                    "phone": {
                        "type": "string",
                        "description": "手机号码",
                    },
                    "aliases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "别名列表（可选）",
                    },
                },
                "required": ["name", "phone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_contact",
            "description": "从通讯录中删除联系人",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "要删除的联系人姓名",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如'北京'、'上海'",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "查询当前日期和时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "设置一个倒计时定时器/闹钟",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": "倒计时秒数",
                    },
                    "label": {
                        "type": "string",
                        "description": "定时器标签，如'煮面'、'提醒开会'",
                    },
                },
                "required": ["seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_timer",
            "description": "取消正在运行的定时器",
            "parameters": {
                "type": "object",
                "properties": {
                    "timer_id": {
                        "type": "string",
                        "description": "定时器ID（可选，不提供则取消最近的）",
                    },
                },
            },
        },
    },
]
