import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff

# Title
st.title("🎓 Student Performance Predictor")
st.write("Predict whether a student will **Pass** or **Fail** using ML")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("student_mat.csv", sep=";")
    return data

data = load_data()
st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# Define Pass/Fail column
# You can adjust threshold (e.g., pass if G3 >= 10)
data['Pass'] = data['G3'].apply(lambda x: 1 if x >= 10 else 0)

# Features and target
X = data.drop(columns=["G1", "G2", "G3", "Pass"])
y = data["Pass"]

# Convert categorical variables to dummies
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
st.subheader("✅ Model Performance")
st.write(f"Accuracy: **{acc:.2f}**")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=["Fail", "Pass"],
    y=["Fail", "Pass"],
    colorscale="Blues",
    showscale=True
)
st.plotly_chart(fig_cm)

# Feature importance
st.subheader("📈 Feature Importance")
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig_imp = px.bar(importances.head(10), x=importances.head(10).values, y=importances.head(10).index, orientation='h')
st.plotly_chart(fig_imp)

# User Prediction
st.subheader("🔮 Try Predicting for a Student")
user_input = {}
for col in X.columns:
    if X[col].nunique() == 2:  # binary categorical
        user_input[col] = st.selectbox(f"{col}", [0, 1])
    else:
        user_input[col] = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))

user_df = pd.DataFrame([user_input])
prediction = model.predict(user_df)[0]

if st.button("Predict Result"):
    st.success("✅ Student is likely to PASS" if prediction == 1 else "❌ Student is likely to FAIL")
