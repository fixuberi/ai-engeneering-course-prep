"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class IrisFeatures(BaseModel):
    """Measurements of one iris flower, in centimetres."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
            ]
        },
    )

    sepal_length: float = Field(gt=0, le=50, description="Sepal length, cm")
    sepal_width: float = Field(gt=0, le=50, description="Sepal width, cm")
    petal_length: float = Field(gt=0, le=50, description="Petal length, cm")
    petal_width: float = Field(gt=0, le=50, description="Petal width, cm")


class PredictResponse(BaseModel):
    """Predicted class plus the model's probability for every class."""

    class_id: int = Field(description="Index of the predicted class")
    class_name: str = Field(description="Name of the predicted class")
    probabilities: dict[str, float] = Field(description="Probability per class name")


class SimilarityRequest(BaseModel):
    """Two vectors of equal length to compare."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "examples": [{"vector_a": [1.0, 2.0, 3.0], "vector_b": [2.0, 4.0, 6.0]}]
        },
    )

    vector_a: list[float] = Field(min_length=1, max_length=10_000, description="First vector")
    vector_b: list[float] = Field(min_length=1, max_length=10_000, description="Second vector")

    @model_validator(mode="after")
    def check_vectors(self):
        if len(self.vector_a) != len(self.vector_b):
            raise ValueError(
                "vector_a and vector_b must have the same length "
                f"(got {len(self.vector_a)} and {len(self.vector_b)})"
            )
        if not any(self.vector_a) or not any(self.vector_b):
            raise ValueError(
                "vectors must not be all zeros: cosine similarity is undefined for a zero vector"
            )
        return self


class SimilarityResponse(BaseModel):
    """Cosine similarity of the two vectors."""

    cosine_similarity: float = Field(description="Cosine similarity, value in [-1, 1]")
    dimension: int = Field(description="Length of each vector")
