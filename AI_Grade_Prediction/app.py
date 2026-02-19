import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("🎓 AI ทำนายผลการเรียน")

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
