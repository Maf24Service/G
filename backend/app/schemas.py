from pydantic import BaseModel, Field


class VisualLandmark(BaseModel):
    box_pixels: list[int] = Field(
        ...,
        description="Bounding box as [x1, y1, x2, y2] in pixel coordinates.",
        min_length=4,
        max_length=4,
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    label: str = "landmark"


class TargetPoint(BaseModel):
    center_x: int
    center_y: int
    radius: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    label: str = "buried_target"


class SceneTag(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class GeoPrediction(BaseModel):
    action_required: str
    predicted_lat: float | None = None
    predicted_lon: float | None = None
    distance_meters: float | None = None


class AnalyzeResponse(BaseModel):
    visual_landmarks: list[VisualLandmark]
    target_points: list[TargetPoint] = Field(default_factory=list)
    geo_prediction: GeoPrediction
    scene_tags: list[SceneTag] = Field(default_factory=list)
    place_summary: str | None = None


class UploadResponse(BaseModel):
    status: str
    file: str


class TrainResponse(BaseModel):
    status: str
    detail: str
    job_id: str


class TrainStatusResponse(BaseModel):
    job_id: str | None = None
    state: str
    detail: str
    images: int | None = None
    labeled_targets: int | None = None
