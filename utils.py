def calculate_grade(score):
    score = float(score)
    if score >= 70:
        return "A", "Pass"
    elif score >= 60:
        return "B", "Pass"
    elif score >= 50:
        return "C", "Pass"
    elif score >= 45:
        return "D", "Pass"
    elif score >= 40:
        return "E", "Pass"
    else:
        return "F", "Fail"


def compute_semester_cgpa(results):
    """Compute CGPA for a semester"""
    if not results:
        return 0.0
    
    total_quality_points = 0
    total_credit_units = 0
    
    for result in results:
        grade_point = get_grade_point(result.grade)
        total_quality_points += grade_point * result.credit_unit
        total_credit_units += result.credit_unit
    
    if total_credit_units == 0:
        return 0.0
    
    return total_quality_points / total_credit_units

def get_grade_point(grade):
    """Convert grade to grade points"""
    grade_points = {
        'A': 5.0,
        'B': 4.0,
        'C': 3.0,
        'D': 2.0,
        'E': 1.0,
        'F': 0.0
    }
    return grade_points.get(grade, 0.0)

def is_passing_grade(grade):
    """Check if a grade is passing (E and above are passing grades)"""
    return grade in ['A', 'B', 'C', 'D', 'E']