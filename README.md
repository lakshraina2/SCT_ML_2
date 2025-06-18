# Customer Segmentation using K-Means Clustering

This project was completed as part of a Machine Learning internship at **SkillCraft Technologies**.

## 🧠 Objective
To apply K-Means Clustering for grouping customers based on:
- Annual Income (k$)
- Spending Score (1–100)

## 📁 Dataset
Dataset used: `Mall_Customers.csv`

## ⚙️ Tools and Technologies
- Python
- Pandas
- Matplotlib
- Scikit-learn

## 🚀 Steps Performed
1. Loaded and explored the dataset.
2. Selected relevant features.
3. Standardized the data using `StandardScaler`.
4. Applied the Elbow Method to determine optimal clusters (k = 5).
5. Built and applied the K-Means model.
6. Visualized the clustered customer segments.

## 📊 Results
The model successfully segmented customers into 5 distinct groups based on purchasing behavior.

## 📎 How to Run
```bash
pip install -r requirements.txt
python KMeans_Customer_Clustering.py
