Decoding Phone Usage Patterns in India# Decoding-Phone-Usage-Patterns-in-India
📱 Decoding Phone Usage Patterns in India

📌 Project Overview

This project analyzes mobile device usage and user behavior in India using machine learning and clustering techniques. The goal is to classify primary phone usage and segment users based on their mobile activity patterns. The project includes a Jupyter Notebook for data analysis and model training, and a Streamlit app for an interactive user experience.

🛠️ Technologies Used

Python for data analysis and machine learning

Jupyter Notebook for EDA, feature engineering, and model training

Scikit-learn & XGBoost for classification models

K-Means, DBSCAN, Gaussian Mixture Models for clustering

Streamlit for building the web-based dashboard

Pandas & NumPy for data manipulation

Matplotlib & Seaborn for data visualization

📂 Project Structure

Decoding_Phone_Usage/
│── Decoding_Phone_Usage.ipynb   # Jupyter Notebook for EDA & Model Training
│── app.py                      # Streamlit App Code
│── clustering_results.csv        # Processed dataset with clusters
│── knn_model.pkl                 # Trained KNN classification model
│── README.md                     # Documentation
└── requirements.txt              # Required Python packages

🔍 Jupyter Notebook: Data Analysis & Model Training

📌 Steps Performed:

Data Preprocessing:

Handled missing values & standardized formats

Encoded categorical variables

Exploratory Data Analysis (EDA):

Visualized feature distributions & correlations

Identified trends in phone usage behavior

Feature Engineering:

Selected key features for classification

Scaled numerical features using StandardScaler

Model Training & Evaluation:

Trained Logistic Regression, Decision Tree, Random Forest, XGBoost, and KNN models

Used accuracy, precision, and recall as evaluation metrics

Selected KNN as the best model and saved it as knn_model.pkl

Clustering Analysis:

Applied K-Means, DBSCAN, Gaussian Mixture Models, and Spectral Clustering

Identified user segments based on usage patterns

🎛️ Streamlit App: Interactive Dashboard

📌 Features of app.py:

Overview Page:

Displays total users, average screen time, and data usage

Shows Most Common Use based on Primary Use mode

Usage Patterns Page:

Users can select a feature and view its distribution plot

Scatter plots for comparing two selected features

User Classification Page:

Allows user input for predicting Primary Use

Uses the trained KNN model for predictions

User Segmentation Page:

Displays clusters from K-Means, DBSCAN, and other techniques

Shows Most Common Use per cluster

🚀 How to Run the Project

🔹 1️⃣ Install Dependencies

pip install -r requirements.txt

🔹 2️⃣ Run Jupyter Notebook (for analysis & model training)

jupyter notebook Decoding_Phone_Usage.ipynb

🔹 3️⃣ Run Streamlit App

streamlit run app1.py

⚡ Fixes & Improvements Made

✅ Fixed Most Common Use issue by mapping category labels correctly
✅ **Handled missing values & data type issues in **Primary Use
✅ Optimized feature selection for better classification accuracy
✅ Added feature selection dropdowns in the Usage Patterns section
✅ Improved clustering visualization for User Segmentation

📌 Next Steps & Enhancements

🔹 Improve model performance with feature engineering & hyperparameter tuning
🔹 Deploy the app on Streamlit Cloud or AWS/GCP for broader accessibility
🔹 Add more interactive visualizations to compare user behavior trends

🎯 Conclusion

This project successfully provides insights into mobile usage patterns using machine learning and an interactive dashboard. The Streamlit app allows users to explore phone usage trends, classify primary use, and analyze user segments effectively.

