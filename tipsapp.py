import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("e:/desktop2025/archive (10)/tips.csv")

# Title
st.title("Tips Data Analysis Dashboard")

# Sidebar filters
st.sidebar.header("Filter Options")
days = st.sidebar.multiselect("Select Day(s):", options=df['day'].unique(), default=df['day'].unique())
times = st.sidebar.multiselect("Select Time(s):", options=df['time'].unique(), default=df['time'].unique())
sexes = st.sidebar.multiselect("Select Gender(s):", options=df['sex'].unique(), default=df['sex'].unique())
smokers = st.sidebar.multiselect("Select Smoker(s):", options=df['smoker'].unique(), default=df['smoker'].unique())

# Filter data
df_filtered = df[(df['day'].isin(days)) &
                 (df['time'].isin(times)) &
                 (df['sex'].isin(sexes)) &
                 (df['smoker'].isin(smokers))]

# Show data
st.subheader("Filtered Data Preview")
st.dataframe(df_filtered.head())

# Summary statistics
st.subheader("Summary Statistics")
st.write(df_filtered.describe())

# Visualizations
st.subheader("Average Tip by Day")
avg_tip_day = df_filtered.groupby("day")["tip"].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=avg_tip_day, x="day", y="tip", ax=ax)
st.pyplot(fig)

st.subheader("Total Bill vs Tip")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df_filtered, x="total_bill", y="tip", hue="sex", ax=ax2)
st.pyplot(fig2)

# Model training
st.subheader("Tip Prediction Model")
model_df = df_filtered.copy()
le = LabelEncoder()
for col in ["sex", "smoker", "day", "time"]:
    model_df[col] = le.fit_transform(model_df[col])

X = model_df.drop("tip", axis=1)
y = model_df["tip"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
st.write(f"Mean Squared Error: {mse:.2f}")

# User input for prediction
st.subheader("Try Tip Prediction")
total_bill_input = st.number_input("Total Bill", min_value=0.0, value=20.0)
size_input = st.slider("Party Size", 1, 10, 2)
sex_input = st.selectbox("Sex", df["sex"].unique())
smoker_input = st.selectbox("Smoker", df["smoker"].unique())
day_input = st.selectbox("Day", df["day"].unique())
time_input = st.selectbox("Time", df["time"].unique())

input_df = pd.DataFrame({
    "total_bill": [total_bill_input],
    "sex": [le.fit(df["sex"]).transform([sex_input])[0]],
    "smoker": [le.fit(df["smoker"]).transform([smoker_input])[0]],
    "day": [le.fit(df["day"]).transform([day_input])[0]],
    "time": [le.fit(df["time"]).transform([time_input])[0]],
    "size": [size_input]
})

predicted_tip = model.predict(input_df)[0]
st.success(f"Predicted Tip: ${predicted_tip:.2f}")
