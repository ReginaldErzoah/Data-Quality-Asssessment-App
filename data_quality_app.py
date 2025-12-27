import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

# CONSTANTS
Placeholders = ["UNKNOWN", "ERROR", "", " "]
Completeness_Threshold = 90
Validity_Threshold = 90
Uniqueness_Threshold = 95
Accuracy_Threshold = 90

# PAGE CONFIG
st.set_page_config(
    page_title="Data Quality & Error Cluster Dashboard",
    layout="wide"
)

# TITLE & INTRO
st.title("Data Quality & Error Cluster Analysis Dashboard")
st.markdown("""
Explore data completeness, validity, and accuracy across transactions.
Identify recurring error clusters by **Location** and **Payment Method**, 
and track data quality trends over time.
""")

# LOAD DATA FUNCTION
@st.cache_data
def load_data(file_path="dirty_cafe_sales.csv"):
    df = pd.read_csv(file_path)

    # Convert date
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")

    # Replace placeholders with NaN
    df.replace(Placeholders, np.nan, inplace=True)

    # Convert numeric columns safely
    numeric_cols = ["Quantity", "Price Per Unit", "Total Spent"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute valid total flag
    df["valid_total"] = np.isclose(
        df["Total Spent"],
        df["Quantity"] * df["Price Per Unit"],
        atol=0.01,
        equal_nan=False
    )

    # Transaction ID consistency
    id_pattern = r"^TXN_\d{7}$"
    df["Transaction_ID_Valid"] = df["Transaction ID"].apply(lambda x: bool(re.match(id_pattern, str(x))))

    # Flag rows with any data issues
    df["has_error"] = df.isna().any(axis=1) | (~df["valid_total"]) | (~df["Transaction_ID_Valid"])

    return df

# FILE UPLOAD OPTION
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
df = load_data(uploaded_file) if uploaded_file else load_data()

# SIDEBAR FILTERS
st.sidebar.header("Filter Data")
selected_location = st.sidebar.multiselect(
    "Select Location(s):",
    options=df["Location"].dropna().unique(),
    default=df["Location"].dropna().unique()
)
selected_payment = st.sidebar.multiselect(
    "Select Payment Method(s):",
    options=df["Payment Method"].dropna().unique(),
    default=df["Payment Method"].dropna().unique()
)

# Prevent empty filtered dataframe
if not selected_location:
    selected_location = df["Location"].dropna().unique()
if not selected_payment:
    selected_payment = df["Payment Method"].dropna().unique()

filtered_df = df[
    df["Location"].isin(selected_location) &
    df["Payment Method"].isin(selected_payment)
]

# DATA QUALITY METRICS
st.markdown("### Data Quality Overview")

# COMPLETENESS: % of non-missing values
completeness = filtered_df.notna().mean() * 100

# VALIDITY: proper column-level checks
validity = pd.Series(dtype=float)

# Numeric columns: must exist and >=0
numeric_cols = ["Quantity", "Price Per Unit", "Total Spent"]
for col in numeric_cols:
    validity[col] = ((filtered_df[col].notna()) & (filtered_df[col] >= 0)).mean() * 100

# Total consistency
validity["Total Consistency"] = filtered_df["valid_total"].mean() * 100

# Categorical columns (exclude Transaction ID)
categorical_cols = filtered_df.select_dtypes(include="object").columns
for col in categorical_cols:
    if col != "Transaction ID":  # consistency already checks this
        validity[col] = filtered_df[col].notna().mean() * 100

# Other metrics
accuracy_score = filtered_df["valid_total"].mean() * 100
consistency_score = filtered_df["Transaction_ID_Valid"].mean() * 100
error_rate = filtered_df["has_error"].mean() * 100
unique_score = 100 - filtered_df.duplicated().mean() * 100

dq_score = round(
    completeness.mean() * 0.3 +
    validity.mean() * 0.3 +
    accuracy_score * 0.2 +
    consistency_score * 0.2,
    2
)

# Display metrics
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{accuracy_score:.2f}%")
col2.metric("Error Rate", f"{error_rate:.2f}%")
col3.metric("Uniqueness", f"{unique_score:.2f}%")
col4.metric("Trans. ID Consistency", f"{consistency_score:.2f}%")
col5.metric("Data Quality Score", f"{dq_score:.2f}%")

st.divider()

# DATA QUALITY SUMMARY TABLE
dq_summary = pd.DataFrame({
    "Completeness (%)": completeness.round(2),
    "Validity (%)": validity.round(2)
})

with st.expander("View Data Quality Summary Table"):
    st.dataframe(dq_summary.style.format("{:.2f}"))

# ERROR RATE VISUALIZATIONS
# By Payment Method
st.subheader("Error Rate by Payment Method")
error_by_payment = filtered_df.groupby("Payment Method")["has_error"].mean() * 100
fig, ax = plt.subplots()
sns.barplot(x=error_by_payment.values, y=error_by_payment.index, color="steelblue", ax=ax)
ax.set_title("Error Rate by Payment Method")
ax.set_xlabel("Error Rate (%)")
plt.tight_layout()
st.pyplot(fig)

# By Location
st.subheader("Error Rate by Location")
error_by_location = filtered_df.groupby("Location")["has_error"].mean() * 100
fig, ax = plt.subplots()
sns.barplot(x=error_by_location.values, y=error_by_location.index, color="steelblue", ax=ax)
ax.set_title("Error Rate by Location")
ax.set_xlabel("Error Rate (%)")
plt.tight_layout()
st.pyplot(fig)

# Error Cluster Heatmap
st.subheader("Error Cluster Heatmap (Location-Payment Method)")
error_pivot = filtered_df.pivot_table(
    values="has_error",
    index="Location",
    columns="Payment Method",
    aggfunc="mean"
) * 100
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(error_pivot, annot=True, fmt=".1f", cmap="Reds", ax=ax)
ax.set_title("Error Clusters by Location-Payment Method")
plt.tight_layout()
st.pyplot(fig)

# Error Trend Over Time
st.subheader("Error Rate Over Time")
filtered_df["Month"] = filtered_df["Transaction Date"].dt.to_period("M")
error_by_month = filtered_df.groupby("Month")["has_error"].mean() * 100
fig, ax = plt.subplots()
error_by_month.plot(marker="o", color="crimson", ax=ax)
ax.set_title("Error Rate Over Time")
ax.set_xlabel("Month")
ax.set_ylabel("Error Rate (%)")
plt.tight_layout()
st.pyplot(fig)

# RECOMMENDATIONS
st.subheader("Recommendations")
recommendations = []

if completeness.mean() < Completeness_Threshold:
    recommendations.append("High missing values - run imputation or drop highly incomplete columns.")
if validity.mean() < Validity_Threshold:
    recommendations.append("Placeholder or invalid values detected - re-clean categorical fields and check business rules.")
if accuracy_score < Accuracy_Threshold:
    recommendations.append("Cross-field totals do not match - consider using a calculated totals column.")
if unique_score < Uniqueness_Threshold:
    recommendations.append("Duplicate records detected - deduplicate IDs or transactions.")
if consistency_score < 100:
    recommendations.append("Check Transaction ID column for inconsistencies")

if recommendations:
    for rec in recommendations:
        st.markdown(f"- {rec}")
else:
    st.success("No critical data quality issues detected!")

# DOWNLOAD BUTTONS
st.divider()
st.markdown("### Export Options")
csv_all = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Data (CSV)",
    data=csv_all,
    file_name="filtered_cafe_sales.csv",
    mime="text/csv"
)
error_data = filtered_df[filtered_df["has_error"]]
csv_errors = error_data.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Only Error Records (CSV)",
    data=csv_errors,
    file_name="error_records.csv",
    mime="text/csv"
)
