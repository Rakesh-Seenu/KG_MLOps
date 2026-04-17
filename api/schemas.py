"""
api/schemas.py
──────────────────────────────────────────────────────────────────────────────
Pydantic schemas for the FastAPI endpoints.

PrimeKG uses native integer indexes for node referencing.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    """Request body for POST /predict"""

    head_index: int = Field(
        ...,
        description="The source node integer index in PrimeKG",
        example=42,
    )
    relation: str = Field(
        ...,
        description="The relation type to predict over",
        example="indication",
    )
    predicted_type: Optional[str] = Field(
        default=None,
        description="Filter results down to a specific node_type (e.g. 'disease' or 'drug')",
        example="disease"
    )
    top_k: int = Field(
        default=10,
        ge=1,       
        le=100,     
        description="Number of predictions to return",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "head_index": 42,
                "relation": "indication",
                "predicted_type": "disease",
                "top_k": 10,
            }
        }


class PredictionItem(BaseModel):
    """A single link prediction result."""
    node_index: int = Field(..., description="Predicted Entity Node Index")
    score: float = Field(..., description="Prediction confidence score (higher = more likely)")
    rank: int = Field(..., description="Rank of this prediction (1 = highest confidence)")


class PredictResponse(BaseModel):
    """Response from POST /predict"""
    head_index: int
    relation: str
    predictions: list[PredictionItem]
    
    class Config:
        json_schema_extra = {
            "example": {
                "head_index": 42,
                "relation": "indication",
                "predictions": [
                    {
                        "node_index": 10984,
                        "score": 9.34,
                        "rank": 1,
                    }
                ],
            }
        }


class HealthResponse(BaseModel):
    """Response from GET /health"""
    status: str
    model_loaded: bool
    device: str
    n_nodes: int
    n_relations: int
