from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from routers import predictive, insight, nlq

app = FastAPI(
    title="DCG Analytics API",
    description="API pour l'analytique de données DCG",
    version="1.0.0"
)

# Configuration CORS pour autoriser Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routeurs
app.include_router(predictive.router, prefix="/api/predictive", tags=["Predictive Analytics"])
app.include_router(insight.router, prefix="/api/insight", tags=["Insight Generation"])
app.include_router(nlq.router, prefix="/api/nlq", tags=["Natural Language Queries"])

@app.get("/")
async def root():
    return {
        "message": "DCG Analytics API",
        "version": "1.0.0",
        "endpoints": {
            "predictive": "/api/predictive/docs",
            "insight": "/api/insight/docs",
            "nlq": "/api/nlq/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "dcg-analytics-api"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)