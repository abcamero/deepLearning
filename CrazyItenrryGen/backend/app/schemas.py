from pydantic import BaseModel, Field
from typing import List

class TripPreference(BaseModel):
    city: str = Field(..., example="Tokyo")
    days: int = Field(..., example=5, ge=1)
    chaos_level: int = Field(..., example=7, ge=1, le=10)
    interests: List[str] = Field(..., example=["weird attractions", "local food", "nightlife"])

class ItineraryItem(BaseModel):
    day: int
    title: str
    description: str
    location: str
    category: str

class ItineraryResponse(BaseModel):
    destination: str
    total_days: int
    chaos_level: int
    itinerary: List[ItineraryItem]
    notes: str

class ItineraryRequest(BaseModel):
    user_id: str = Field(..., example="user_123")
    preference: TripPreference
