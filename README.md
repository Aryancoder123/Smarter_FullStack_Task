# SPA: HTML Chunk Search (React + FastAPI + Weaviate)

This project implements a single-page app where users submit a website URL and a search query. The backend fetches the page, splits DOM content into token-limited chunks, indexes them into Weaviate, and returns the top-10 matches for the query.

Folders:
- `backend/` - FastAPI application and Docker Compose for Weaviate
- `frontend/` - React + Vite SPA

Quick start (developer machine):

1. Start Weaviate (Docker Compose)

   Open a terminal at the project root and run:

```powershell
docker compose up -d
```

2. Backend (recommended inside a Python venv)

```powershell
cd backend
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the frontend in the browser (Vite will print the URL). The UI posts to `http://localhost:8000/search`.

Notes:
- This README contains run instructions and a local Weaviate docker-compose. See `backend/README.md` for backend-specific notes.
