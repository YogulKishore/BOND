from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.database import init_db
from routers.auth_router import router as auth_router
from routers.chat_router import router as chat_router
from routers.session_router import router as session_router
from routers.profile_router import router as profile_router
from routers.async_router import router as async_router
import os

app = FastAPI(title="BOND API")

_allowed_origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "https://bond-sable.vercel.app",
]
_extra = os.environ.get("ALLOWED_ORIGIN")
if _extra:
    _allowed_origins.append(_extra)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_db()
    print("BOND backend started")

@app.get("/")
async def root():
    return {"status": "BOND API running"}

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(session_router)
app.include_router(profile_router)
app.include_router(async_router)
