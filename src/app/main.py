from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.serving.inference import predict  

# Initialize FastAPI application
app = FastAPI(
    title="Student Burnout Prediction API",
    description="ML API for predicting student burnout levels",
    version="1.0.0"
)

# === HEALTH CHECK ENDPOINT ===
@app.get("/")
def root():
    """
    Health check endpoint for monitoring and load balancer health checks.
    """
    return {"status": "ok"}

# === REQUEST DATA SCHEMA ===
# Pydantic model for automatic validation and API documentation
class StudentData(BaseModel):
    """
    Student data schema for burnout prediction.
    """
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

# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: StudentData):
    """
    Main prediction endpoint for student burnout prediction.
    
    This endpoint:
    1. Receives validated data via Pydantic model
    2. Calls the inference pipeline to transform features and predict
    3. Returns burnout prediction in JSON format
    """
    try:
        # Convert Pydantic model to dict and call inference pipeline
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        # Return error details for debugging (consider logging in production)
        return {"error": str(e)}


# =================================================== # 


# === GRADIO WEB INTERFACE ===
def gradio_interface(
    gender, age, course, year, daily_study_hours, daily_sleep_hours,
    screen_time_hours, stress_level, anxiety_score, depression_score,
    academic_pressure_score, financial_stress_score, social_support_score,
    physical_activity_hours, sleep_quality, attendance_percentage, cgpa, internet_quality
):
    """
    Gradio interface function that processes form inputs and returns prediction.
    """
    data = {
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
    
    # Call same inference pipeline as API endpoint
    result = predict(data)
    return str(result)  # Return as string for Gradio display

# === GRADIO UI CONFIGURATION ===
# Build comprehensive Gradio interface
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(["Male", "Female", "Other"], label="Gender", value="Male"),
        gr.Number(label="Age (years)", value=21),
        gr.Dropdown(["BTech", "BCA", "BSc", "MBA", "BA", "Other"], label="Course", value="BTech", allow_custom_value=True),
        gr.Dropdown(["1st", "2nd", "3rd", "4th", "5th"], label="Year", value="3rd"),
        gr.Number(label="Daily Study Hours", value=4.0),
        gr.Number(label="Daily Sleep Hours", value=7.0),
        gr.Number(label="Screen Time Hours", value=5.0),
        gr.Dropdown(["Low", "Medium", "High"], label="Stress Level", value="Medium"),
        gr.Slider(1, 10, label="Anxiety Score", step=1, value=5),
        gr.Slider(1, 10, label="Depression Score", step=1, value=5),
        gr.Slider(1, 10, label="Academic Pressure Score", step=1, value=6),
        gr.Slider(1, 10, label="Financial Stress Score", step=1, value=5),
        gr.Slider(1, 10, label="Social Support Score", step=1, value=7),
        gr.Number(label="Physical Activity Hours", value=1.0),
        gr.Dropdown(["Poor", "Average", "Good"], label="Sleep Quality", value="Average"),
        gr.Number(label="Attendance Percentage (0-100)", value=85.0),
        gr.Number(label="CGPA (0-10)", value=7.5),
        gr.Dropdown(["Poor", "Average", "Good"], label="Internet Quality", value="Good"),
    ],
    outputs=gr.Textbox(label="Burnout Prediction", lines=2),
    title="🔮 Student Burnout Predictor",
    description="""
    **Predict student burnout probability using machine learning**
    
    Fill in the student details below to get a prediction of their burnout level (Low, Medium, or High).
    """,
    examples=[
        # High burnout risk
        ["Male", 22, "BTech", "4th", 8.0, 4.0, 9.0, "High", 9, 8, 10, 8, 3, 0.0, "Poor", 45.0, 5.5, "Poor"],
        # Low burnout risk  
        ["Female", 20, "BA", "2nd", 3.0, 8.0, 3.0, "Low", 2, 1, 3, 2, 9, 2.0, "Good", 95.0, 9.2, "Good"]
    ],
    theme=gr.themes.Soft() 
)

# === MOUNT GRADIO UI INTO FASTAPI ===
app = gr.mount_gradio_app(
    app,           
    demo,          
    path="/ui"     
)