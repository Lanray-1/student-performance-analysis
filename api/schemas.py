from typing import Literal

from pydantic import BaseModel, Field


class StudentFeatures(BaseModel):
    age: int = Field(..., ge=16, le=28)
    gender: Literal["Female", "Male", "Other"]
    major: Literal["Arts", "Biology", "Business", "Computer Science", "Engineering", "Psychology"]
    study_hours_per_day: float = Field(..., ge=0, le=12)
    social_media_hours: float = Field(..., ge=0, le=5)
    netflix_hours: float = Field(..., ge=0, le=4)
    part_time_job: Literal["No", "Yes"]
    attendance_percentage: float = Field(..., ge=40, le=100)
    sleep_hours: float = Field(..., ge=4, le=12)
    diet_quality: Literal["Fair", "Good", "Poor"]
    exercise_frequency: int = Field(..., ge=0, le=7)
    parental_education_level: Literal["Bachelor", "High School", "Master", "PhD", "Some College"]
    internet_quality: Literal["High", "Low", "Medium"]
    mental_health_rating: float = Field(..., ge=1, le=10)
    extracurricular_participation: Literal["No", "Yes"]
    previous_gpa: float = Field(..., ge=1.64, le=4.0)
    semester: int = Field(..., ge=1, le=8)
    stress_level: float = Field(..., ge=1, le=10)
    social_activity: int = Field(..., ge=0, le=5)
    screen_time: float = Field(..., ge=0.3, le=21)
    study_environment: Literal["Cafe", "Co-Learning Group", "Dorm", "Library", "Quiet Room"]
    access_to_tutoring: Literal["No", "Yes"]
    family_income_range: Literal["High", "Low", "Medium"]
    parental_support_level: int = Field(..., ge=1, le=10)
    motivation_level: int = Field(..., ge=1, le=10)
    exam_anxiety_score: int = Field(..., ge=5, le=10)
    learning_style: Literal["Auditory", "Kinesthetic", "Reading", "Visual"]
    time_management_score: float = Field(..., ge=1, le=10)


class ExamScoreResponse(BaseModel):
    predicted_exam_score: float = Field(..., ge=0, le=100)


class DropoutRiskResponse(BaseModel):
    dropout_probability: float = Field(..., ge=0, le=1)
    dropout_risk: Literal["Yes", "No"]