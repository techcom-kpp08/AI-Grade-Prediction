import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 1️⃣ โหลดข้อมูล
data = pd.read_csv("student_data.csv")

# 2️⃣ แปลง Pass/Fail เป็นตัวเลข
data['final_grade'] = data['final_grade'].map({'Pass': 1, 'Fail': 0})

# 3️⃣ แยก X และ y
X = data[['attendance', 'study_hours', 'assignment_score', 'late_submit']]
y = data['final_grade']

# 4️⃣ แบ่ง Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ สร้างโมเดล
model = RandomForestClassifier(random_state=42)

# 6️⃣ Train โมเดล
model.fit(X_train, y_train)

# 7️⃣ ทดสอบโมเดล
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8️⃣ บันทึกโมเดล (ต้องอยู่ล่างสุด)
joblib.dump(model, "model.pkl")
print("บันทึกโมเดลเรียบร้อยแล้ว")
