#  Comprehensive Predictive Modeling & Machine Learning Demo  
### **Breast Cancer Classification · Diabetes Regression · Clustering Analysis**

This project is a **full machine learning demonstration suite** that walks through **classification**, **regression**, **clustering**, **model comparison**, **evaluation**, and **visualization** using real-world datasets.

It is built to be presentation-ready for **IEEE-style research**, coursework, or data science portfolios.

---

##  Features

✔ **Classification (Supervised Learning)**  
• Breast Cancer Dataset (Scikit-Learn)  
• Logistic Regression, Random Forest, SVM, KNN  
• F1, Precision, Recall, ROC-AUC  
• Confusion matrix, ROC curves, feature importance  

✔ **Regression**  
• Diabetes Dataset  
• Linear Regression, Decision Tree, Random Forest  
• MAE, MSE, RMSE, R²  
• Residual plots, predicted vs actual  

✔ **Clustering (Unsupervised Learning)**  
• K-Means, DBSCAN, Hierarchical  
• Silhouette, Calinski-Harabasz, Davies-Bouldin  
• Exhaustive parameter tuning (e.g., DBSCAN eps/min_samples)  
• Dendrograms, cluster visualization, noise detection  

✔ **Automatic EDA**  
• Heatmaps  
• Distribution plots  
• Summary statistics  
• Missing value detection  

✔ **Advanced Visualizations**  
• Comparative model scoring  
• Elbow method  
• Clustering metric comparison  

✔ **SHAP-ready** (optional)  
• Enables model interpretability if installed

---

##  Project Structure

2. Install Dependencies

Your script uses:

Python 3.8+

pandas

numpy

matplotlib

seaborn

scikit-learn

scipy

shap (optional)

Install with requirements file:

pip install -r requirements.txt


Or manually:

pip install pandas numpy matplotlib seaborn scikit-learn scipy shap

▶️ How to Run the Program

Run:

python "Breast Cancer Classification.py"


The script will:

Print Python and library diagnostics

Load all datasets

Run:

Classification demo

Regression demo

Clustering demo

Display all plots

Print best models and summaries

📊 Classification: Breast Cancer Dataset

The script trains and evaluates 4 models:

Model	Metrics Used
Logistic Regression	Accuracy, Precision, Recall, F1, ROC-AUC
Random Forest	Feature importances + full metrics
SVM	Full classification metrics
KNN	k=5 neighbor classifier

Includes:

F1-score comparison

Confusion matrix

ROC curves

Feature importance bars

📈 Regression: Diabetes Dataset

Models included:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Metrics computed:

MAE

MSE

RMSE

R²

Visualizations:

R² comparison

Predicted vs actual scatter

Residual plot

MAE comparison

🧩 Clustering: K-Means, DBSCAN, Hierarchical

Clustering is run on:

Simple synthetic dataset

Complex/noisy dataset

Iris dataset

Algorithms and metrics:

Method	Metrics
K-Means	Silhouette, Calinski-Harabasz, Davies-Bouldin
DBSCAN	Noise detection, cluster count, silhouette
Hierarchical	Ward linkage, dendrogram, silhouette

Also includes:

Elbow method

Silhouette scoring across k

DBSCAN parameter search

Full metric comparison charts

🛠️ Parameter Search & Optimization
DBSCAN Grid Search

eps = [0.2, 0.3, 0.4, 0.5]

min_samples = [5, 10, 15]

K-Means Search

k = 2 → 10

Elbow + Silhouette scoring

Automatically selects optimal k

📘 Summary of Capabilities
1. Classification

Multiple models

ROC-AUC

Feature importance

2. Regression

Regression error metrics

Diagnostic visualizations

3. Clustering

Visual + statistical cluster comparison

Noise detection

Dendrograms

4. Visualization

Dozens of automatic plots

5. Reproducibility

Fully self-contained and ready for academic/IEEE publication.
