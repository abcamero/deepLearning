import os
import uvicorn

if __name__ == "__main__":
    os.environ.setdefault("USE_SAMPLE_OUTPUT", "1")
    print("Starting Crazy Itinerary Generator on http://127.0.0.1:8000")
    print("Sample itinerary mode enabled. No external API keys required.")
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
