from fastapi import FastAPI, HTTPException

from api.schemas import DropoutRiskResponse, ExamScoreResponse, StudentFeatures
from src.prediction import predict_dropout_risk, predict_exam_score

app = FastAPI(
    title="Student Success Analytics Platform",
    description="Predicts exam score and dropout risk from student behavioral and demographic features.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict/exam-score", response_model=ExamScoreResponse)
def predict_exam_score_endpoint(features: StudentFeatures):
    try:
        score = predict_exam_score(features.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return ExamScoreResponse(predicted_exam_score=score)


@app.post("/predict/dropout-risk", response_model=DropoutRiskResponse)
def predict_dropout_risk_endpoint(features: StudentFeatures):
    try:
        result = predict_dropout_risk(features.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return DropoutRiskResponse(**result)