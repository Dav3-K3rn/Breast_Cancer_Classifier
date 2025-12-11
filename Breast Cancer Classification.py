import sys
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("IEEE PREDICTIVE MODELING DEMO - COMPREHENSIVE VERSION")
print("=" * 70)

print(f"Python version: {sys.version}")
print()

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn import metrics
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.datasets import load_breast_cancer, load_diabetes, make_blobs, load_iris
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import fcluster
    
    try:
        import shap
        SHAP_AVAILABLE = True
        print("SHAP installed - Model interpretation enabled")
    except ImportError:
        SHAP_AVAILABLE = False
        print("SHAP not installed - Install with: pip install shap")
    
    print("All required packages imported successfully")
    
except ImportError as e:
    print(f"✗ Missing required package: {e}")
    print("Please install with: pip install pandas numpy matplotlib seaborn scikit-learn scipy")
    sys.exit(1)

plt.style.use('default')
sns.set_palette("husl")

def load_standard_datasets():
    print("\nLoading standard datasets for analysis...")
    
    cancer_data = load_breast_cancer()
    diabetes_data = load_diabetes()
    iris_data = load_iris()
    
    df_cancer = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
    df_cancer['diagnosis'] = cancer_data.target
    df_cancer['diagnosis'] = df_cancer['diagnosis'].map({0: 'malignant', 1: 'benign'})
    
    df_diabetes = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
    df_diabetes['target'] = diabetes_data.target
    
    X_cluster1, y_cluster1 = make_blobs(n_samples=300, centers=3, n_features=2, 
                                       random_state=42, cluster_std=1.2)
    df_cluster1 = pd.DataFrame(X_cluster1, columns=['Feature_1', 'Feature_2'])
    df_cluster1['true_cluster'] = y_cluster1
    
    df_iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df_iris['species'] = iris_data.target
    df_iris['species_name'] = df_iris['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    X_complex, _ = make_blobs(n_samples=250, centers=[[2, 2], [-2, -2], [2, -2]], 
                             cluster_std=[0.5, 0.8, 1.2], random_state=42)
    np.random.seed(42)
    noise = np.random.uniform(low=-6, high=6, size=(50, 2))
    X_complex = np.vstack([X_complex, noise])
    y_complex = np.concatenate([np.zeros(250), -1 * np.ones(50)])
    
    df_complex = pd.DataFrame(X_complex, columns=['Feature_1', 'Feature_2'])
    df_complex['true_cluster'] = y_complex
    
    return {
        'classification': df_cancer,
        'regression': df_diabetes, 
        'clustering_simple': df_cluster1,
        'clustering_iris': df_iris,
        'clustering_complex': df_complex
    }

datasets = load_standard_datasets()

def perform_basic_eda(df, dataset_name):
    print(f"\n" + "="*50)
    print(f"EXPLORATORY DATA ANALYSIS: {dataset_name.upper()}")
    print("="*50)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    
    print(f"\nBasic statistics:")
    print(df.describe())
    
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"\nMissing values detected: {missing_values}")
    
    plt.figure(figsize=(12, 4))
    
    if 'diagnosis' in df.columns:
        plt.subplot(1, 3, 1)
        df['diagnosis'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
        plt.title('Target Distribution')
        plt.xlabel('Diagnosis')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
    elif 'target' in df.columns:
        plt.subplot(1, 3, 1)
        plt.hist(df['target'], bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.title('Target Distribution')
        plt.xlabel('Target Value')
        plt.ylabel('Frequency')
    elif 'true_cluster' in df.columns:
        plt.subplot(1, 3, 1)
        unique_clusters = len(df['true_cluster'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, unique_clusters))
        df['true_cluster'].value_counts().sort_index().plot(
            kind='bar', color=colors[:unique_clusters])
        plt.title('True Cluster Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
    elif 'species' in df.columns:
        plt.subplot(1, 3, 1)
        df['species_name'].value_counts().plot(kind='bar', color=['red', 'green', 'blue'])
        plt.title('Species Distribution')
        plt.xlabel('Species')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr_cols = min(10, len(numeric_df.columns))
        corr_matrix = numeric_df.iloc[:, :corr_cols].corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, square=True)
        plt.title(f'Correlation Heatmap (First {corr_cols} features)')
    
    plt.subplot(1, 3, 3)
    if len(numeric_df.columns) > 0:
        plt.hist(numeric_df.iloc[:, 0], bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.title(f'Distribution of {numeric_df.columns[0]}')
        plt.xlabel(numeric_df.columns[0])
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return df

def run_classification_demo():
    print("\n" + "="*50)
    print("CLASSIFICATION MODELING - Breast Cancer Dataset")
    print("="*50)
    
    df = datasets['classification']
    df_clean = perform_basic_eda(df, "Breast Cancer Classification")
    
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'K-NN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        roc_auc = metrics.roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    model_names = list(results.keys())
    f1_scores = [results[name]['f1'] for name in model_names]
    
    bars = plt.bar(range(len(model_names)), f1_scores, 
                   color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Model Comparison (F1-Scores)', fontweight='bold')
    plt.xlabel('Models')
    plt.ylabel('F1-Score')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        plt.text(i, bar.get_height() + 0.01, f'{score:.3f}', 
                 ha='center', va='bottom', fontweight='bold')
    
    plt.subplot(1, 4, 2)
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']
    y_pred_best = best_model.predict(X_test_scaled)
    
    cm = metrics.confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'], 
                yticklabels=['Benign', 'Malignant'])
    plt.title(f'Confusion Matrix\n{best_model_name}', fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.subplot(1, 4, 3)
    for name, result in results.items():
        y_pred_proba = result['model'].predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison', fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 4)
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[-10:]
        
        plt.barh(range(len(indices)), importances[indices], color='steelblue')
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Features (Random Forest)', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n  Best Model: {best_model_name} (F1-Score: {results[best_model_name]['f1']:.4f})")
    
    return results

def run_regression_demo():
    print("\n" + "="*50)
    print("REGRESSION MODELING - Diabetes Dataset")
    print("="*50)
    
    df = datasets['regression']
    df_clean = perform_basic_eda(df, "Diabetes Regression")
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"  MAE:  {mae:.4f}")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    
    bars = plt.bar(range(len(model_names)), r2_scores, 
                   color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('Model Comparison (R² Scores)', fontweight='bold')
    plt.xlabel('Models')
    plt.ylabel('R² Score')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    
    for i, (bar, score) in enumerate(zip(bars, r2_scores)):
        plt.text(i, bar.get_height() + 0.01, f'{score:.3f}', 
                 ha='center', va='bottom', fontweight='bold')
    
    plt.subplot(1, 4, 2)
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]['model']
    y_pred_best = best_model.predict(X_test_scaled)
    
    plt.scatter(y_test, y_pred_best, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual\n{best_model_name}', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 3)
    residuals = y_test - y_pred_best
    plt.scatter(y_pred_best, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot\n{best_model_name}', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 4)
    # FIXED: Create proper boxplot data
    mae_values = [[results[m]['mae']] for m in results.keys()]  # List of lists format
    plt.boxplot(mae_values, 
                labels=list(results.keys()))
    plt.title('MAE Comparison Across Models', fontweight='bold')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n  Best Model: {best_model_name} (R²: {results[best_model_name]['r2']:.4f})")
    
    return results

def compare_clustering_parameters():
    print("\n" + "="*50)
    print("CLUSTERING PARAMETER TUNING COMPARISON")
    print("="*50)
    
    df = datasets['clustering_complex']
    X = df[['Feature_1', 'Feature_2']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    print("\n--- DBSCAN Parameter Grid Search ---")
    eps_values = [0.2, 0.3, 0.4, 0.5]
    min_samples_values = [5, 10, 15]
    
    best_dbscan_score = -1
    best_dbscan_params = None
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            
            n_clusters = len(np.unique(labels[labels != -1]))
            noise_points = np.sum(labels == -1)
            
            if n_clusters > 1 and len(labels[labels != -1]) > 0:
                mask = labels != -1
                if np.sum(mask) > 1:
                    silhouette = metrics.silhouette_score(X_scaled[mask], labels[mask])
                    if silhouette > best_dbscan_score:
                        best_dbscan_score = silhouette
                        best_dbscan_params = (eps, min_samples)
                    
                    print(f"  eps={eps:.1f}, min_samples={min_samples}: "
                          f"Clusters={n_clusters}, Noise={noise_points}, "
                          f"Silhouette={silhouette:.4f}")
    
    if best_dbscan_params:
        print(f"\n✓ Best DBSCAN: eps={best_dbscan_params[0]}, "
              f"min_samples={best_dbscan_params[1]} (Silhouette: {best_dbscan_score:.4f})")
    
    print("\n--- Hierarchical Linkage Methods ---")
    linkages = ['ward', 'complete', 'average', 'single']
    
    for linkage_method in linkages:
        hierarchical = AgglomerativeClustering(n_clusters=3, linkage=linkage_method)
        labels = hierarchical.fit_predict(X_scaled)
        silhouette = metrics.silhouette_score(X_scaled, labels)
        print(f"  {linkage_method}: Silhouette={silhouette:.4f}")
    
    print("\n--- K-Means Optimal K Determination ---")
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(metrics.silhouette_score(X_scaled, labels))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different k', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    best_k = K_range[np.argmax(silhouette_scores)]
    print(f"\n✓ Optimal k based on silhouette: {best_k} "
          f"(Score: {np.max(silhouette_scores):.4f})")
    
    return {
        'best_dbscan': best_dbscan_params,
        'best_k': best_k,
        'silhouette_scores': silhouette_scores
    }

def run_clustering_demo():
    print("\n" + "="*50)
    print("COMPREHENSIVE CLUSTERING ANALYSIS")
    print("="*50)
    
    print("\n=== Simple Dataset (Well-Separated Clusters) ===")
    df_simple = datasets['clustering_simple']
    perform_basic_eda(df_simple, "Simple Clustering Dataset")
    
    print("\n=== Complex Dataset (With Noise) ===")
    df_complex = datasets['clustering_complex']
    perform_basic_eda(df_complex, "Complex Clustering Dataset")
    
    print("\n=== Iris Dataset (Real-World) ===")
    df_iris = datasets['clustering_iris']
    perform_basic_eda(df_iris, "Iris Dataset")
    
    df = df_complex
    X = df[['Feature_1', 'Feature_2']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    print("\n--- K-Means Clustering ---")
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    silhouette_kmeans = metrics.silhouette_score(X_scaled, kmeans_labels)
    calinski_kmeans = metrics.calinski_harabasz_score(X_scaled, kmeans_labels)
    davies_kmeans = metrics.davies_bouldin_score(X_scaled, kmeans_labels)
    
    results['K-Means'] = {
        'model': kmeans,
        'labels': kmeans_labels,
        'silhouette': silhouette_kmeans,
        'calinski': calinski_kmeans,
        'davies': davies_kmeans,
        'n_clusters': len(np.unique(kmeans_labels))
    }
    print(f"  Silhouette Score: {silhouette_kmeans:.4f}")
    print(f"  Calinski-Harabasz: {calinski_kmeans:.2f}")
    print(f"  Davies-Bouldin: {davies_kmeans:.4f}")
    print(f"  Number of clusters found: {len(np.unique(kmeans_labels))}")
    
    print("\n--- DBSCAN Clustering ---")
    dbscan = DBSCAN(eps=0.3, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    n_clusters_dbscan = len(np.unique(dbscan_labels[dbscan_labels != -1]))
    noise_points = np.sum(dbscan_labels == -1)
    
    if n_clusters_dbscan > 1:
        mask = dbscan_labels != -1
        if np.sum(mask) > 1:
            silhouette_dbscan = metrics.silhouette_score(X_scaled[mask], dbscan_labels[mask])
            calinski_dbscan = metrics.calinski_harabasz_score(X_scaled[mask], dbscan_labels[mask])
            davies_dbscan = metrics.davies_bouldin_score(X_scaled[mask], dbscan_labels[mask])
        else:
            silhouette_dbscan = -1
            calinski_dbscan = -1
            davies_dbscan = -1
    else:
        silhouette_dbscan = -1
        calinski_dbscan = -1
        davies_dbscan = -1
    
    results['DBSCAN'] = {
        'model': dbscan,
        'labels': dbscan_labels,
        'silhouette': silhouette_dbscan,
        'calinski': calinski_dbscan,
        'davies': davies_dbscan,
        'n_clusters': n_clusters_dbscan,
        'noise_points': noise_points
    }
    print(f"  Number of clusters found: {n_clusters_dbscan}")
    print(f"  Noise points: {noise_points}")
    if silhouette_dbscan != -1:
        print(f"  Silhouette Score: {silhouette_dbscan:.4f}")
        print(f"  Calinski-Harabasz: {calinski_dbscan:.2f}")
        print(f"  Davies-Bouldin: {davies_dbscan:.4f}")
    
    print("\n--- Hierarchical Clustering ---")
    hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    silhouette_hierarchical = metrics.silhouette_score(X_scaled, hierarchical_labels)
    calinski_hierarchical = metrics.calinski_harabasz_score(X_scaled, hierarchical_labels)
    davies_hierarchical = metrics.davies_bouldin_score(X_scaled, hierarchical_labels)
    
    results['Hierarchical'] = {
        'model': hierarchical,
        'labels': hierarchical_labels,
        'silhouette': silhouette_hierarchical,
        'calinski': calinski_hierarchical,
        'davies': davies_hierarchical,
        'n_clusters': len(np.unique(hierarchical_labels))
    }
    print(f"  Silhouette Score: {silhouette_hierarchical:.4f}")
    print(f"  Calinski-Harabasz: {calinski_hierarchical:.2f}")
    print(f"  Davies-Bouldin: {davies_hierarchical:.4f}")
    print(f"  Number of clusters: {len(np.unique(hierarchical_labels))}")
    
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 3, 1)
    scatter = plt.scatter(df['Feature_1'], df['Feature_2'], 
                         c=df['true_cluster'], cmap='tab20', alpha=0.7, s=50)
    plt.colorbar(scatter)
    plt.title('True Clusters (Ground Truth)', fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    scatter = plt.scatter(df['Feature_1'], df['Feature_2'], 
                         c=kmeans_labels, cmap='tab20', alpha=0.7, s=50)
    plt.colorbar(scatter)
    plt.title(f'K-Means Clustering\nSilhouette: {silhouette_kmeans:.3f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', 
                s=200, alpha=0.8, label='Centroids')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    cmap = plt.cm.tab20
    colors = cmap(dbscan_labels % cmap.N)
    colors[dbscan_labels == -1] = [0.5, 0.5, 0.5, 1.0]
    
    scatter = plt.scatter(df['Feature_1'], df['Feature_2'], 
                         c=colors, alpha=0.7, s=50)
    plt.title(f'DBSCAN Clustering\nClusters: {n_clusters_dbscan}, Noise: {noise_points}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 4)
    scatter = plt.scatter(df['Feature_1'], df['Feature_2'], 
                         c=hierarchical_labels, cmap='tab20', alpha=0.7, s=50)
    plt.colorbar(scatter)
    plt.title(f'Hierarchical Clustering\nSilhouette: {silhouette_hierarchical:.3f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    linked = linkage(X_scaled, method='ward')
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True,
               truncate_mode='lastp',
               p=20,
               show_contracted=True)
    plt.title('Hierarchical: Dendrogram', fontsize=14, fontweight='bold')
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')
    plt.axhline(y=10, color='r', linestyle='--', alpha=0.7, label='Cut line')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    metrics_data = {
        'K-Means': [results['K-Means']['silhouette'], 
                   results['K-Means']['calinski']/100 if results['K-Means']['calinski'] > 0 else 0,
                   1/results['K-Means']['davies'] if results['K-Means']['davies'] > 0 else 0],
        'DBSCAN': [results['DBSCAN']['silhouette'] if results['DBSCAN']['silhouette'] != -1 else 0,
                  results['DBSCAN']['calinski']/100 if results['DBSCAN']['calinski'] != -1 else 0,
                  1/results['DBSCAN']['davies'] if results['DBSCAN']['davies'] != -1 and results['DBSCAN']['davies'] > 0 else 0],
        'Hierarchical': [results['Hierarchical']['silhouette'],
                        results['Hierarchical']['calinski']/100,
                        1/results['Hierarchical']['davies'] if results['Hierarchical']['davies'] > 0 else 0]
    }
    
    x = np.arange(len(metrics_data))
    width = 0.25
    
    for i, (metric_name, color) in enumerate(zip(['Silhouette', 'Calinski', 'Davies-Inverse'], 
                                                ['skyblue', 'lightgreen', 'lightcoral'])):
        values = [metrics_data[algo][i] for algo in metrics_data.keys()]
        plt.bar(x + i*width - width, values, width, label=metric_name, color=color)
    
    plt.xlabel('Algorithm')
    plt.ylabel('Score (Normalized)')
    plt.title('Algorithm Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, metrics_data.keys())
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("CLUSTERING ALGORITHM COMPARISON")
    print("="*60)
    print(f"{'Algorithm':<20} {'Clusters':<10} {'Silhouette':<12} {'Calinski':<12} {'Davies':<12} {'Noise':<10}")
    print("-"*60)
    
    for algo, result in results.items():
        silhouette = result['silhouette']
        calinski = result['calinski']
        davies = result['davies']
        
        silhouette_str = f"{silhouette:.4f}" if silhouette != -1 else "N/A"
        calinski_str = f"{calinski:.2f}" if calinski != -1 else "N/A"
        davies_str = f"{davies:.4f}" if davies != -1 and davies > 0 else "N/A"
        
        noise = result.get('noise_points', 0)
        
        print(f"{algo:<20} {result['n_clusters']:<10} {silhouette_str:<12} "
              f"{calinski_str:<12} {davies_str:<12} {noise:<10}")
    
    valid_results = {k: v for k, v in results.items() if v['silhouette'] != -1}
    if valid_results:
        best_algo = max(valid_results.keys(), 
                       key=lambda x: valid_results[x]['silhouette'])
        print(f"\n✓ Best Algorithm: {best_algo} "
              f"(Silhouette: {valid_results[best_algo]['silhouette']:.4f})")
    
    param_results = compare_clustering_parameters()
    
    return results, param_results

def main():
    print("\n" + "="*70)
    print("STARTING COMPREHENSIVE PREDICTIVE MODELING DEMONSTRATION")
    print("="*70)
    
    try:
        classification_results = run_classification_demo()
        regression_results = run_regression_demo()
        clustering_results, param_results = run_clustering_demo()
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nSUMMARY OF RESULTS:")
    print("1. Classification: Breast Cancer Diagnosis")
    print("   - Multiple algorithms compared (Logistic Regression, Random Forest, SVM, K-NN)")
    print("   - Best model determined by F1-Score")
    print("\n2. Regression: Diabetes Progression Prediction")
    print("   - Linear Regression, Decision Tree, Random Forest models")
    print("   - Best model determined by R² Score")
    print("\n3. Clustering: Comprehensive Unsupervised Learning")
    print("   - K-Means, DBSCAN, and Hierarchical Clustering")
    print("   - Multiple evaluation metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)")
    print("   - Parameter tuning and comparison")
    print("\n4. Key Features:")
    print("   - Comprehensive visualizations for all analyses")
    print("   - Multiple datasets for different scenarios")
    print("   - Complete evaluation framework")
    print("\nThis comprehensive demo provides a complete framework for IEEE research paper.")
    print("All code is reproducible, well-documented, and ready for academic publication.")

if __name__ == "__main__":
    main()