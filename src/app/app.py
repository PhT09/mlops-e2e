from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.serving.inference import predict

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

class StudentData(BaseModel):
    gender: str
    age: float
    course: str
    year: str
    daily_study_hours: float
    daily_sleep_hours: float
    screen_time_hours: float
    stress_level: str
    anxiety_score: float
    depression_score: float
    academic_pressure_score: float
    financial_stress_score: float
    social_support_score: float
    physical_activity_hours: float
    sleep_quality: str
    attendance_percentage: float
    cgpa: float
    internet_quality: str

@app.post("/predict")
def api_predict(data: StudentData):
    try:
        out = predict(data.dict())
        return {"prediction": out}
    except Exception as e:
        return {"error": str(e)}

# --- Gradio UI wrappers ---
def gradio_interface(
    gender, age, course, year, daily_study_hours, daily_sleep_hours,
    screen_time_hours, stress_level, anxiety_score, depression_score,
    academic_pressure_score, financial_stress_score, social_support_score,
    physical_activity_hours, sleep_quality, attendance_percentage, cgpa, internet_quality
):
    payload = {
        "gender": gender,
        "age": float(age),
        "course": course,
        "year": year,
        "daily_study_hours": float(daily_study_hours),
        "daily_sleep_hours": float(daily_sleep_hours),
        "screen_time_hours": float(screen_time_hours),
        "stress_level": stress_level,
        "anxiety_score": float(anxiety_score),
        "depression_score": float(depression_score),
        "academic_pressure_score": float(academic_pressure_score),
        "financial_stress_score": float(financial_stress_score),
        "social_support_score": float(social_support_score),
        "physical_activity_hours": float(physical_activity_hours),
        "sleep_quality": sleep_quality,
        "attendance_percentage": float(attendance_percentage),
        "cgpa": float(cgpa),
        "internet_quality": internet_quality,
    }
    out = predict(payload)
    return str(out)

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(["Male", "Female", "Other"], label="Gender"),
        gr.Number(label="Age (years)"),
        gr.Dropdown(["BTech", "BCA", "BSc", "MBA", "BA", "Other"], label="Course", allow_custom_value=True),
        gr.Dropdown(["1st", "2nd", "3rd", "4th", "5th"], label="Year"),
        gr.Number(label="Daily Study Hours"),
        gr.Number(label="Daily Sleep Hours"),
        gr.Number(label="Screen Time Hours"),
        gr.Dropdown(["Low", "Medium", "High"], label="Stress Level"),
        gr.Slider(1, 10, label="Anxiety Score", step=1),
        gr.Slider(1, 10, label="Depression Score", step=1),
        gr.Slider(1, 10, label="Academic Pressure Score", step=1),
        gr.Slider(1, 10, label="Financial Stress Score", step=1),
        gr.Slider(1, 10, label="Social Support Score", step=1),
        gr.Number(label="Physical Activity Hours"),
        gr.Dropdown(["Poor", "Average", "Good"], label="Sleep Quality"),
        gr.Number(label="Attendance Percentage (0-100)"),
        gr.Number(label="CGPA (0-10)"),
        gr.Dropdown(["Poor", "Average", "Good"], label="Internet Quality"),
    ],
    outputs="text",
    title="Student Burnout Predictor",
    description="Fill in the student details to get a burnout prediction.",
)

app = gr.mount_gradio_app(app, demo, path="/ui")