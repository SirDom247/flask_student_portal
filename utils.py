def calculate_grade(score):
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
    total_points = 0
    for r in results:
        # Assign points based on grade
        grade_points = {'A':5,'B':4,'C':3,'D':2,'E':1,'F':0}
        total_points += grade_points.get(r.grade,0)
    return round(total_points / len(results), 2) if results else 0.0
