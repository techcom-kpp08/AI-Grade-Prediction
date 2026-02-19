import pandas as pd
import random

data = []

for i in range(300):   # สร้าง 300 แถว
    attendance = random.randint(50, 100)
    study_hours = random.randint(0, 5)
    assignment_score = random.randint(40, 100)
    late_submit = random.randint(0, 5)

    # เงื่อนไขตัดสินเกรด
    if attendance > 75 and assignment_score > 70 and late_submit < 2:
        final_grade = "Pass"
    else:
        final_grade = "Fail"

    data.append([
        attendance,
        study_hours,
        assignment_score,
        late_submit,
        final_grade
    ])

df = pd.DataFrame(data, columns=[
    "attendance",
    "study_hours",
    "assignment_score",
    "late_submit",
    "final_grade"
])

df.to_csv("student_data.csv", index=False)

print("สร้าง dataset เสร็จแล้ว!")
