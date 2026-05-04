import os
import httpx
from ..core.config import settings

async def request_ai_itinerary(payload: dict) -> dict:
    if os.getenv("USE_SAMPLE_OUTPUT", "false").lower() in ("1", "true", "yes"):
        return sample_itinerary(payload)

    if not settings.llm_api_url or not settings.llm_api_key:
        return sample_itinerary(payload)

    prompt = build_prompt(payload)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            settings.llm_api_url,
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            json={"prompt": prompt, "max_tokens": 700},
        )
        response.raise_for_status()
        data = response.json()

    return parse_llm_response(data, payload)

def build_prompt(payload: dict) -> str:
    interests = ", ".join(payload["interests"])
    return (
        f"Generate a crazy travel itinerary for {payload['destination']} over {payload['days']} days. "
        f"The chaos level is {payload['chaos_level']} on a scale of 1 to 10. "
        f"Interests: {interests}. "
        "Use external travel and local discovery data to create a surprising multi-day plan. "
        "Return a JSON object with itinerary items and notes."
    )

def parse_llm_response(data: dict, payload: dict) -> dict:
    if "itinerary" in data:
        return data
    return sample_itinerary(payload)


def sample_itinerary(payload: dict) -> dict:
    destination = payload.get("destination", "Unknown")
    days = int(payload.get("days", 3))
    interests = payload.get("interests", [])
    chaos = payload.get("chaos_level", 5)

    items = [
        {
            "day": i + 1,
            "title": f"Day {i + 1}: {destination} Chaos Tour",
            "description": (
                f"Experience {destination}'s most surprising spots with a chaos level of {chaos}. "
                f"Enjoy {interests[i % len(interests)] if interests else 'local curiosities'} and unexpected detours."
            ),
            "location": destination,
            "category": "adventure",
        }
        for i in range(days)
    ]

    return {
        "itinerary": items,
        "notes": "This sample itinerary is generated locally without calling an external LLM API.",
    }
