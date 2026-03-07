"""Bayes-learner API server."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Bayes-learner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from bayes_learner.server.routes import health
app.include_router(health.router)

@app.on_event("startup")
async def startup():
    print("\n📡 Bayes-learner API Routes:")
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            methods = ", ".join(route.methods - {"HEAD", "OPTIONS"})
            if methods:
                print(f"  {methods:8} {route.path}")
    print()

@app.get("/")
async def root():
    return {"status": "ok", "service": "bayes-learner"}