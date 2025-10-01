%%writefile app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load dataset ---
carematch = pd.read_csv("carematch_requests.csv")

# --- Dashboard Title ---
st.title("📊 Carematch Dashboard")

# --- Show sample data ---
st.subheader("Sample Data")
st.write(carematch.head())

# --- Sidebar filter ---
bins = st.sidebar.slider("Select number of bins for histogram", 5, 50, 20)
# --- Descriptive Stats for All Variables ---
st.header("📊 Descriptive Statistics (All Variables)")

# Descriptive stats for numeric + categorical variables
desc_stats = carematch.describe(include="all").T   # transpose for readability

# Show as interactive table
st.dataframe(desc_stats)
