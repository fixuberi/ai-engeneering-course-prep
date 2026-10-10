"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, ConfigDict, Field


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
