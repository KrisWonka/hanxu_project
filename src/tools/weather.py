"""
Weather query via wttr.in (free, no API key needed).
"""

from __future__ import annotations

import logging
import urllib.request
import urllib.error
import json

logger = logging.getLogger(__name__)

WTTR_URL = "https://wttr.in/{city}?format=j1&lang=zh"
TIMEOUT = 10


def get_weather(city: str = "北京") -> dict:
    """
    Fetch current weather for a city. Returns a simplified dict.
    Falls back to an error dict on failure.
    """
    url = WTTR_URL.format(city=urllib.request.quote(city))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode())

        current = data["current_condition"][0]
        area = data.get("nearest_area", [{}])[0]
        area_name = area.get("areaName", [{}])[0].get("value", city)

        return {
            "success": True,
            "city": area_name,
            "temp_c": current.get("temp_C", "?"),
            "feels_like_c": current.get("FeelsLikeC", "?"),
            "humidity": current.get("humidity", "?"),
            "description": current.get("lang_zh", [{}])[0].get("value", current.get("weatherDesc", [{}])[0].get("value", "")),
            "wind_speed_kmph": current.get("windspeedKmph", "?"),
        }
    except Exception as e:
        logger.error("Weather query failed for %s: %s", city, e)
        return {"success": False, "error": f"查询天气失败: {e}"}
