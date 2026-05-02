from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str
    surname: str
    patronymic: str | None = None


class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    name: str
    surname: str
    patronymic: str | None
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class TokenPairResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


PredictionMode = Literal["standard", "risk_group", "aggressive"]
DiabetesClass = Literal[0, 1, 2]
PredictionInputValue = float | int | None
PredictionComputedValue = float


class FactorItem(BaseModel):
    feature: str
    label: str
    value: float
    normal_range: str
    shap_value: float


class PredictionRequest(BaseModel):
    name: str
    mode: PredictionMode
    weight_kg: float = Field(
        ...,
        description="Масса тела, кг. Сохраняется в user_input_data и нужна для смены режима.",
    )
    RIDAGEYR: int
    RIAGENDR: int
    BMXHT: float
    BMXWAIST: float
    hypertension: int
    LBXTC: float
    LBDHDD: float
    LBXGLU: float
    LBXTR: float | None = None
    DMDEDUC2: float
    LBXSAT: float | None = None
    LBXSASS: float | None = None
    LBXSUA: float | None = None
    LBXVIDMS: float | None = None
    LBXSBU: float | None = None
    URXUMA: float | None = None
    LBXHGB: float | None = None
    LBXRBCSI: float | None = None
    LBXSTB: float | None = None
    LBXSAL: float | None = None
    LBXSAPSI: float | None = None
    LBXWBCSI: float | None = None
    LBXLYPCT: float | None = None
    LBXMCVSI: float | None = None
    LBXRDW: float | None = None
    LBXCRP: float | None = None
    MCQ300C: int
    MCQ160A: int | None = None
    MCQ160E: int | None = None
    MCQ160F: int | None = None
    HSD010: float | None = None
    LBXSCR: float = Field(
        ...,
        description="Креатинин сыворотки (LBXSCR), в единицах пайплайна. Нужен для расчёта и смены режима.",
    )


class PredictionResponse(BaseModel):
    id: UUID
    name: str
    mode: PredictionMode
    user_input_data: dict[str, PredictionInputValue]
    computed_data: dict[str, PredictionComputedValue] = Field(
        ...,
        description=(
            "Показатели, посчитанные на бэкенде после импутации (до скейлера): ИМТ, eGFR, WHtR, "
            "TyG_Index, ALT_AST_Ratio, Log_CRP, Log_SUA, PhenoAge_accel. "
            "Произведения вроде Age×глюкоза в ответ не включаются, но участвуют в модели."
        ),
    )
    has_feedback: bool
    predicted_class: DiabetesClass
    class_label: str
    probabilities: dict[str, float]
    top_factors: list[FactorItem]
    created_at: str


class PredictionListItemResponse(BaseModel):
    id: UUID
    name: str
    mode: PredictionMode
    predicted_class: DiabetesClass
    class_label: str
    probabilities: dict[str, float]
    top_factors: list[FactorItem]
    created_at: str


class PredictionFeedbackRequest(BaseModel):
    real_class: DiabetesClass
    comment: str | None = None


class PredictionModeUpdateRequest(BaseModel):
    mode: PredictionMode
