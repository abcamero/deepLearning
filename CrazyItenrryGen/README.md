# Crazy Itinerary Generator

A full-stack travel itinerary generator scaffold built around the architecture shown in the design diagram.

This project contains:
- `backend/` — Python FastAPI service with itinerary generation, external API placeholders, and database-ready models.
- `frontend/` — Next.js + Tailwind CSS interface for collecting user travel preferences and displaying itineraries.

## Getting Started

### Backend

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r backend/requirements.txt
   ```
2. Copy variables from `backend/.env.example` into `backend/.env` and fill your keys.
3. Run the backend:
   ```bash
   uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend

1. Install dependencies:
   ```bash
   cd CrazyItenrryGen/frontend
   npm install
   ```
2. Start the app:
   ```bash
   npm run dev
   ```

## Architecture

- Frontend: Next.js + Tailwind CSS
- Backend: FastAPI
- Database: SQLAlchemy-ready models (SQLite default, PostgreSQL / Docker ready)
- External APIs: Geocoding/hotels/flights service placeholders
- AI engine: LLM prompt and response orchestration placeholder
