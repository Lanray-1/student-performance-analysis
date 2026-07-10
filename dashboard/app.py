import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("Student Success Analytics Platform")
st.write("Predict exam score and dropout risk from student features.")

with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=16, max_value=28, value=22)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        major = st.selectbox("Major", ["Arts", "Biology", "Business", "Computer Science", "Engineering", "Psychology"])
        study_hours_per_day = st.slider("Study hours per day", 0.0, 12.0, 4.0)
        social_media_hours = st.slider("Social media hours", 0.0, 5.0, 2.5)
        netflix_hours = st.slider("Netflix hours", 0.0, 4.0, 2.0)
        part_time_job = st.selectbox("Part-time job", ["No", "Yes"])
        attendance_percentage = st.slider("Attendance %", 40.0, 100.0, 70.0)
        sleep_hours = st.slider("Sleep hours", 4.0, 12.0, 7.0)
        diet_quality = st.selectbox("Diet quality", ["Fair", "Good", "Poor"])
        exercise_frequency = st.slider("Exercise frequency (days/week)", 0, 7, 3)
        parental_education_level = st.selectbox(
            "Parental education level", ["Bachelor", "High School", "Master", "PhD", "Some College"]
        )
        internet_quality = st.selectbox("Internet quality", ["High", "Low", "Medium"])
        mental_health_rating = st.slider("Mental health rating", 1.0, 10.0, 6.8)

    with col2:
        extracurricular_participation = st.selectbox("Extracurricular participation", ["No", "Yes"])
        previous_gpa = st.slider("Previous GPA", 1.64, 4.0, 3.6)
        semester = st.number_input("Semester", min_value=1, max_value=8, value=4)
        stress_level = st.slider("Stress level", 1.0, 10.0, 5.0)
        social_activity = st.slider("Social activity", 0, 5, 2)
        screen_time = st.slider("Screen time (hrs)", 0.3, 21.0, 9.7)
        study_environment = st.selectbox(
            "Study environment", ["Cafe", "Co-Learning Group", "Dorm", "Library", "Quiet Room"]
        )
        access_to_tutoring = st.selectbox("Access to tutoring", ["No", "Yes"])
        family_income_range = st.selectbox("Family income range", ["High", "Low", "Medium"])
        parental_support_level = st.slider("Parental support level", 1, 10, 5)
        motivation_level = st.slider("Motivation level", 1, 10, 5)
        exam_anxiety_score = st.slider("Exam anxiety score", 5, 10, 8)
        learning_style = st.selectbox("Learning style", ["Auditory", "Kinesthetic", "Reading", "Visual"])
        time_management_score = st.slider("Time management score", 1.0, 10.0, 5.5)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "age": age, "gender": gender, "major": major,
        "study_hours_per_day": study_hours_per_day, "social_media_hours": social_media_hours,
        "netflix_hours": netflix_hours, "part_time_job": part_time_job,
        "attendance_percentage": attendance_percentage, "sleep_hours": sleep_hours,
        "diet_quality": diet_quality, "exercise_frequency": exercise_frequency,
        "parental_education_level": parental_education_level, "internet_quality": internet_quality,
        "mental_health_rating": mental_health_rating,
        "extracurricular_participation": extracurricular_participation, "previous_gpa": previous_gpa,
        "semester": semester, "stress_level": stress_level, "social_activity": social_activity,
        "screen_time": screen_time, "study_environment": study_environment,
        "access_to_tutoring": access_to_tutoring, "family_income_range": family_income_range,
        "parental_support_level": parental_support_level, "motivation_level": motivation_level,
        "exam_anxiety_score": exam_anxiety_score, "learning_style": learning_style,
        "time_management_score": time_management_score,
    }

    try:
        score_resp = requests.post(f"{API_URL}/predict/exam-score", json=payload, timeout=5)
        risk_resp = requests.post(f"{API_URL}/predict/dropout-risk", json=payload, timeout=5)
        score_resp.raise_for_status()
        risk_resp.raise_for_status()

        score = score_resp.json()["predicted_exam_score"]
        risk = risk_resp.json()

        col1, col2 = st.columns(2)
        col1.metric("Predicted Exam Score", f"{score:.1f}")
        col2.metric("Dropout Risk", risk["dropout_risk"], f"{risk['dropout_probability']:.2%} probability")

    except requests.exceptions.ConnectionError:
        st.error("Can't reach the API. Is it running? (`python -m uvicorn api.main:app --reload`)")
    except requests.exceptions.HTTPError as e:
        st.error(f"API returned an error: {e}")