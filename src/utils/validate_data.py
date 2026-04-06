import great_expectations as ge
from typing import Tuple, List


def validate_student_data(df) -> Tuple[bool, List[str]]:
    print("Starting data validation...")
    failed_expectations = []
    
    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   Validating schema and required columns...")
    required_cols = [
        "student_id", "age", "gender", "course", "year", 
        "daily_study_hours", "daily_sleep_hours", "screen_time_hours", "physical_activity_hours",
        "stress_level", "anxiety_score", "depression_score", "academic_pressure_score",
        "financial_stress_score", "social_support_score", "burnout_level",
        "sleep_quality", "attendance_percentage", "cgpa", "internet_quality"
    ]
    for col in required_cols:
        if col not in df.columns:
            failed_expectations.append(f"column_to_exist:{col}")
    
    if "student_id" in df.columns and df["student_id"].isnull().any():
        failed_expectations.append("values_to_not_be_null:student_id")
        
    # === BUSINESS LOGIC VALIDATION ===
    if "gender" in df.columns and not df["gender"].isin(["Male", "Female", "Other"]).all():
        failed_expectations.append("values_to_be_in_set:gender")

    # === NUMERIC RANGE VALIDATION ===
    if "age" in df.columns and not df["age"].between(15, 60).all():
        failed_expectations.append("values_to_be_between:age")
        
    for col in ["daily_study_hours", "daily_sleep_hours", "screen_time_hours", "physical_activity_hours"]:
        if col in df.columns and not df[col].between(0, 24).all():
            failed_expectations.append(f"values_to_be_between:{col}")
            
    for col in ["anxiety_score", "depression_score", "academic_pressure_score", "financial_stress_score", "social_support_score"]:
        if col in df.columns and not df[col].between(1, 10).all():
            failed_expectations.append(f"values_to_be_between:{col}")
            
    if "attendance_percentage" in df.columns and not df["attendance_percentage"].between(0, 100).all():
        failed_expectations.append("values_to_be_between:attendance_percentage")
        
    if "cgpa" in df.columns and not df["cgpa"].between(0, 10).all():
        failed_expectations.append("values_to_be_between:cgpa")
        
    # === STATISTICAL VALIDATION ===
    if "burnout_level" in df.columns and df["burnout_level"].isnull().any():
        failed_expectations.append("values_to_not_be_null:burnout_level")
    if "cgpa" in df.columns and df["cgpa"].isnull().any():
        failed_expectations.append("values_to_not_be_null:cgpa")
        
    success = len(failed_expectations) == 0
    
    if success:
        print("✅ Data validation PASSED")
    else:
        print(f"❌ Data validation FAILED: {len(failed_expectations)} checks failed")
        print(f"   Failed expectations: {failed_expectations}")
    
    return success, failed_expectations