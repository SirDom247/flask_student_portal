# fix_data.py
from app import app, db
from models import Result
import pandas as pd

def fix_existing_data():
    with app.app_context():
        print("Checking for problematic data...")
        
        # Find records with decimal academic_session
        decimal_sessions = Result.query.filter(Result.academic_session.contains('.')).all()
        print(f"Found {len(decimal_sessions)} records with decimal academic_session")
        
        # Find records with problematic semester
        decimal_semesters = Result.query.filter(Result.semester.contains('.')).all()
        print(f"Found {len(decimal_semesters)} records with decimal semester")
        
        # Fix them
        for result in decimal_sessions:
            result.academic_session = '2023/2024'
        
        for result in decimal_semesters:
            result.semester = '1'
        
        # Fix CGPA
        all_results = Result.query.all()
        for result in all_results:
            result.semester_cgpa = round(result.semester_cgpa, 2)
        
        db.session.commit()
        print("Data fixed successfully!")

def test_excel_upload():
    """Test if Excel upload works correctly"""
    print("\nTesting Excel upload...")
    # You can create a test Excel file here or use an existing one

if __name__ == "__main__":
    fix_existing_data()