from fastapi import APIRouter, HTTPException
from ..schemas import ItineraryRequest, ItineraryResponse
from ..services.itinerary import build_itinerary

router = APIRouter(prefix="/api", tags=["itineraries"])

@router.post("/itineraries", response_model=ItineraryResponse)
async def create_itinerary(request: ItineraryRequest):
    itinerary = await build_itinerary(request)
    if not itinerary:
        raise HTTPException(status_code=500, detail="Unable to generate itinerary")
    return itinerary
