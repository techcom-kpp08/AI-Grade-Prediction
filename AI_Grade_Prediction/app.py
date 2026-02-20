import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import os
import joblib

# โหลดโมเดลที่ train ไว้แล้ว
model = joblib.load("model.pkl")
data = pd.read_csv("student_data.csv")
data['final_grade'] = data['final_grade'].map({'Pass': 1, 'Fail': 0})

st.title("🎓 AI ทำนายผลการเรียน")
st.write("กรอกข้อมูลพฤติกรรมการเรียนเพื่อทำนายผล")

attendance = st.slider("Attendance (%)", 0, 100, 75)
study_hours = st.slider("Study Hours per Day", 0, 10, 2)
assignment_score = st.slider("Assignment Score", 0, 100, 70)
late_submit = st.slider("Late Submission Count", 0, 10, 1)

if st.button("Predict"):
    input_data = pd.DataFrame([[attendance, study_hours, assignment_score, late_submit]],
                              columns=["attendance","study_hours","assignment_score","late_submit"])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success("✅ มีแนวโน้มผ่าน")
    else:
        st.error("⚠️ มีความเสี่ยงตก")

    st.write("ความน่าจะเป็นผ่าน:", round(probability[0][1]*100,2), "%")

st.subheader("📊 สัดส่วนผลการเรียน")

fig1, ax1 = plt.subplots()
data['final_grade'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_xticklabels(['Fail','Pass'], rotation=0)
st.pyplot(fig1)

st.subheader("📈 ความสำคัญของตัวแปร")

feature_importance = model.feature_importances_

fig2, ax2 = plt.subplots()
ax2.bar(["attendance","study_hours","assignment_score","late_submit"],
        feature_importance)
ax2.set_ylabel("Importance Score")
st.pyplot(fig2)

st.subheader("📉 Attendance vs Grade")

fig3, ax3 = plt.subplots()
sns.boxplot(x=data['final_grade'], y=data['attendance'], ax=ax3)
ax3.set_xticklabels(['Fail','Pass'])
st.pyplot(fig3)

if os.path.exists("model.pkl"):
    model = joblib.load("model.pkl")
else:
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd
    
    data = pd.read_csv("student_data.csv")
    X = data[['study_hours', 'attendance']]
    y = data['grade']
    
    model = RandomForestRegressor()
    model.fit(X, y)
    
    joblib.dump(model, "model.pkl")