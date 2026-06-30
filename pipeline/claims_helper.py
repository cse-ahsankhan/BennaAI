"""
Benna AI — Claims & Delay Analysis Helper

Extracts date/location metadata from queries and pulls verified weather (Open-Meteo)
and public holiday (Nager.Date) data to support claims analysis.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
import urllib.parse
from datetime import datetime
from typing import Dict, Any, Optional

import config

logger = logging.getLogger(__name__)

# Major GCC Cities coordinates mapping
GCC_CITIES = {
    "dubai": {"lat": 25.2048, "lon": 55.2708, "country": "AE"},
    "abu dhabi": {"lat": 24.4539, "lon": 54.3773, "country": "AE"},
    "riyadh": {"lat": 24.7136, "lon": 46.6753, "country": "SA"},
    "jeddah": {"lat": 21.4858, "lon": 39.1925, "country": "SA"},
    "doha": {"lat": 25.2854, "lon": 51.5310, "country": "QA"},
    "manama": {"lat": 26.2285, "lon": 50.5860, "country": "BH"},
    "muscat": {"lat": 23.5859, "lon": 58.4059, "country": "OM"},
    "kuwait": {"lat": 29.3759, "lon": 47.9774, "country": "KW"},
    "kuwait city": {"lat": 29.3759, "lon": 47.9774, "country": "KW"},
}

_EXTRACTION_SYSTEM = """\
You are a metadata extraction assistant for a construction document intelligence system.
Your job is to identify if the user query is asking about a delay or claims event that involves weather, holidays, or specific dates.

If the query is related to weather/delays/claims/events on a specific date and/or location:
1. "is_claim_query": true
2. "date": Extract the date in "YYYY-MM-DD" format. If only a year/month is mentioned, estimate the first day or leave it. If not found, use null.
3. "city": Target city name (in English, e.g. "Dubai"). If not found, use null.
4. "country_code": 2-letter ISO country code (e.g., AE, SA, QA, OM, KW, BH). If not found, use null.

If the query is NOT related to any specific dates/events/delays (e.g., "What are the payment terms?", "Who is the engineer?"):
1. "is_claim_query": false
2. "date": null
3. "city": null
4. "country_code": null

Respond ONLY with a valid JSON object. Do not include markdown codeblocks (like ```json), introduction, or explanation.

Example:
Query: "Was the delay on Oct 12, 2025 due to weather in Dubai justified?"
Output: {"is_claim_query": true, "date": "2025-10-12", "city": "Dubai", "country_code": "AE"}
"""


def _call_llm_direct(provider: str, system: str, user: str) -> str:
    """Send a direct prompt to the active LLM provider, bypassing RAG formatting."""
    if provider == "claude":
        import anthropic

        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    elif provider == "ollama":
        try:
            from langchain_community.llms import Ollama
        except ImportError:
            raise ImportError("langchain-community is required for Ollama support")

        llm = Ollama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
            num_ctx=2048,
            num_predict=256,
        )
        prompt = f"<<SYS>>\n{system}\n<</SYS>>\n\n{user}\n\nResponse:"
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            if "connection" in str(exc).lower() or "refused" in str(exc).lower():
                raise ConnectionError(
                    f"Cannot connect to Ollama at {config.OLLAMA_BASE_URL}."
                ) from exc
            raise
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'")


def _parse_json(text: str) -> Dict[str, Any]:
    """Parse json robustly from model output, handling potential code blocks or wrappers."""
    cleaned = text.strip()
    # Strip markdown code blocks if any
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback to regex finding of first JSON-like block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def _geocode_city(city: str) -> Optional[Dict[str, Any]]:
    """Geocode a city name to get lat, lon, and country code."""
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Benna-AI-App"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            results = data.get("results")
            if results:
                return {
                    "lat": results[0]["latitude"],
                    "lon": results[0]["longitude"],
                    "country": results[0].get("country_code", "AE").upper(),
                }
    except Exception as exc:
        logger.warning("Geocoding failed for city %s: %s", city, exc)
    return None


def _fetch_weather(lat: float, lon: float, date_str: str) -> Optional[Dict[str, Any]]:
    """Fetch daily weather metrics from Open-Meteo."""
    try:
        target_dt = datetime.strptime(date_str, "%Y-%m-%d")
        is_past = (datetime.now() - target_dt).days >= 2
    except ValueError:
        return None

    if is_past:
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max&timezone=auto"
    else:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max&timezone=auto"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Benna-AI-App"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            daily = data.get("daily", {})
            if daily and "temperature_2m_max" in daily:
                max_temp = daily["temperature_2m_max"][0]
                precip = daily["precipitation_sum"][0]
                max_wind = daily["wind_speed_10m_max"][0]

                units = data.get("daily_units", {})
                return {
                    "max_temp": max_temp if max_temp is not None else 0.0,
                    "max_temp_unit": units.get("temperature_2m_max", "°C"),
                    "precipitation": precip if precip is not None else 0.0,
                    "precipitation_unit": units.get("precipitation_sum", "mm"),
                    "max_wind": max_wind if max_wind is not None else 0.0,
                    "max_wind_unit": units.get("wind_speed_10m_max", "km/h"),
                }
    except Exception as exc:
        logger.warning("Weather fetch failed: %s", exc)
    return None


def _fetch_holiday(country_code: str, date_str: str) -> Optional[Dict[str, Any]]:
    """Fetch public holidays for a country and check if date is a holiday."""
    try:
        year = date_str.split("-")[0]
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
        req = urllib.request.Request(url, headers={"User-Agent": "Benna-AI-App"})
        with urllib.request.urlopen(req, timeout=5) as response:
            content = response.read().decode()
            if not content.strip():
                return None
            holidays = json.loads(content)
            for h in holidays:
                if h.get("date") == date_str:
                    return {
                        "is_holiday": True,
                        "name": h.get("name", "Public Holiday"),
                        "local_name": h.get("localName", "Public Holiday"),
                    }
    except Exception as exc:
        logger.warning("Holiday fetch failed: %s", exc)
    return None


def analyze_query_for_claim(query: str, provider: str) -> Optional[Dict[str, Any]]:
    """
    Analyzes query, fetches weather & holiday metadata if claim query.
    
    Returns structured data or None if not relevant.
    """
    try:
        response_text = _call_llm_direct(provider, _EXTRACTION_SYSTEM, f"Query: {query}")
        meta = _parse_json(response_text)
    except Exception as exc:
        logger.warning("Claim metadata extraction failed: %s", exc)
        return None

    if not meta.get("is_claim_query") or not meta.get("date"):
        return None

    date_str = meta["date"]
    city = meta.get("city") or "Dubai"  # Default to Dubai if not specified in GCC context
    country_code = meta.get("country_code") or "AE"

    # Resolve coordinates
    coords = GCC_CITIES.get(city.lower())
    if not coords:
        resolved = _geocode_city(city)
        if resolved:
            coords = resolved
            country_code = resolved["country"]
        else:
            coords = GCC_CITIES["dubai"]  # Fallback to Dubai coords

    # Fetch weather and holiday
    weather = _fetch_weather(coords["lat"], coords["lon"], date_str)
    holiday = _fetch_holiday(country_code, date_str)

    # Weekday check
    try:
        target_dt = datetime.strptime(date_str, "%Y-%m-%d")
        weekday = target_dt.strftime("%A")
        is_weekend = target_dt.weekday() in (5, 6)  # Saturday & Sunday
    except Exception:
        weekday = "Unknown"
        is_weekend = False

    return {
        "date": date_str,
        "city": city,
        "country_code": country_code,
        "weather": weather,
        "holiday": holiday,
        "weekday": weekday,
        "is_weekend": is_weekend,
    }


def format_claims_context(data: Dict[str, Any]) -> str:
    """Format extracted claim weather/holiday data as standard context text."""
    weather = data.get("weather")
    holiday = data.get("holiday")

    parts = [
        f"--- EXTERNAL VERIFIED DATA FOR DATE: {data['date']} ---",
        f"Location: {data['city']}, {data['country_code']}",
        f"Day of Week: {data['weekday']} ({'Weekend' if data['is_weekend'] else 'Weekday'})",
    ]

    if weather:
        parts.append(
            f"Weather: Max Temp {weather['max_temp']}{weather['max_temp_unit']}, "
            f"Precipitation {weather['precipitation']}{weather['precipitation_unit']}, "
            f"Max Wind Speed {weather['max_wind']}{weather['max_wind_unit']}"
        )
    else:
        parts.append("Weather: Data unavailable")

    if holiday:
        parts.append(f"Public Holiday: Yes - {holiday['name']} ({holiday['local_name']})")
    else:
        parts.append("Public Holiday: No")

    parts.append("--- END OF EXTERNAL VERIFIED DATA ---")
    return "\n".join(parts)
