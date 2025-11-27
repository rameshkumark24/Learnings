# MODEL DEVELOPMENT & MODEL EVALUATION – FULL NOTES

## PART 1 — MODEL DEVELOPMENT (Full Pipeline)

Model development means the entire workflow from data loading → to training → to deployment.
Here is the correct real-world + academic structure.

### 1. Problem Definition

Understand:
- **Type of ML task**
  - Classification (spam/ham, fraud detection)
  - Regression (house price prediction)
  - Clustering
  - NLP, CV
- **Target variable (y)**
- **Business goals**
- **Evaluation metrics required**

### 2. Data Collection

Methods:
- Through CSV files
- Databases
- APIs
- Web scraping
- Sensors, logs
- Cloud storage

Store data with:
- Pandas
- SQL
- Data lakes (for big companies)

### 3. Data Understanding (EDA)

This step finds:
- Structure
- Missing values
- Outliers
- Feature types
- Class imbalance
- Distribution patterns

Tools:
- Pandas
- Matplotlib
- Seaborn

### 4. Data Preprocessing

Preprocessing depends on your data.

**4.1 Handling Missing Values**
- Mean/median (numerical)
- Mode (categorical)
- Forward/backward fill
- Drop row/column

**4.2 Outlier Treatment**
- IQR method
- Z-score
- Domain-based rules

**4.3 Categorical Encoding**
- Label Encoding
- One-hot Encoding
- Target Encoding (advanced)

**4.4 Feature Scaling**

Algorithms that need scaling:
- KNN
- SVM
- Logistic Regression
- Neural Networks
- PCA

Techniques:
- StandardScaler
- MinMaxScaler
- RobustScaler

**4.5 Train-Test Split**
