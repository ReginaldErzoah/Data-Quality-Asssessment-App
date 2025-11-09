# Data Quality Assessment Project

## Project Overview
This project is an interactive Streamlit dashboard that analyzes and visualizes data quality metrics and error clusters in transactional datasets.  
It helps identify issues in completeness, validity, and accuracy — providing quick insights into where and how data quality problems occur.

For a live demo, check the streamlit app (Here)[].

The project also demonstrates the use of:
- Python for data preprocessing  
- Pandas & NumPy for quality checks  
- Seaborn & Matplotlib for visual insights  
- Streamlit for interactive visualization and data exploration  

---
## Use Case
This project is ideal for:
- Data Analysts validating data pipelines
- BI professionals monitoring data quality in reports
- Organizations seeking proactive error tracking in transaction systems

 --- 
## Features
**Data Quality Metrics**
- Completeness and Validity per field  
- Overall Accuracy and Error Rate summary  

**Interactive Filtering**
- Filter data by **Location** and **Payment Method**  

**Visual Error Insights**
- Error rate by Payment Method  
- Error rate by Location  
- Error Cluster Heatmap (Location × Payment Method)  
- Error Rate Trend over Time  

**Data Exports**
- Download filtered dataset (CSV)  
- Download only error records (CSV)  

---

## Dataset
The sample dataset used in this project is `dirty_cafe_sales.csv` (in project folder), a simulated transactional dataset containing:
- Transaction Date  
- Location  
- Payment Method  
- Quantity  
- Price Per Unit  
- Total Spent  

It intentionally includes missing, invalid, and inconsistent values to demonstrate real-world data quality issues.

---

## Tech Stack

| Category | Tools |
|-----------|-------|
| Programming | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |
| Notebook Analysis | Jupyter Notebook |

---


