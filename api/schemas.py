"""
api/schemas.py
──────────────────────────────────────────────────────────────────────────────
Pydantic schemas for the FastAPI endpoints.

🎓 WHAT IS PYDANTIC?
  Pydantic validates data using Python type hints.
  - Automatically validates request JSON against the schema
  - Returns friendly error messages if input is wrong
  - Generates JSON Schema → FastAPI uses this for /docs auto-documentation
  - Industry standard for Python API development
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    """Request body for POST /predict"""

    head_entity: str = Field(
        ...,
        description="The source entity in the knowledge graph",
        example="Disease::MESH:D003920",
    )
    relation: str = Field(
        ...,
        description="The relation type to predict over",
        example="GNBR::T::Compound:Disease",
    )
    top_k: int = Field(
        default=10,
        ge=1,       # ge = greater than or equal to
        le=100,     # le = less than or equal to
        description="Number of predictions to return",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "head_entity": "Disease::MESH:D003920",
                "relation": "GNBR::T::Compound:Disease",
                "top_k": 10,
            }
        }


class PredictionItem(BaseModel):
    """A single link prediction result."""
    entity: str = Field(..., description="Predicted entity name")
    entity_type: str = Field(..., description="Entity type (Gene, Disease, Compound, etc.)")
    score: float = Field(..., description="Prediction confidence score (higher = more likely)")
    rank: int = Field(..., description="Rank of this prediction (1 = highest confidence)")


class PredictResponse(BaseModel):
    """Response from POST /predict"""
    head_entity: str
    relation: str
    predictions: list[PredictionItem]
    
    class Config:
        json_schema_extra = {
            "example": {
                "head_entity": "Disease::MESH:D003920",
                "relation": "GNBR::T::Compound:Disease",
                "predictions": [
                    {
                        "entity": "Compound::DB00678",
                        "entity_type": "Compound",
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
    n_entities: int
    n_relations: int
