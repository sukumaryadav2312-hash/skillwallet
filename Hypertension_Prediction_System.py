

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# create directory for saved plots
os.makedirs('plots', exist_ok=True)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
import json as json_lib

print("All libraries imported successfully!")
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# If trained model already exists, inform and exit early
if os.path.exists('best_hypertension_model.pkl') and os.path.exists('label_encoder.pkl'):
    print("Existing model artifacts detected. To retrain delete the .pkl files or modify the script.")

    import sys
    sys.exit(0)

# ---
# 2. Data Collection & Preparation
# 2.1 Load the Dataset

df = pd.read_csv(r"C:\Users\sunny\Downloads\patient_data.csv")
print(f"Dataset Shape: {df.shape}")
print(f"Records: {df.shape[0]}, Features: {df.shape[1]}")
df.head()

# 2.2 Data Inspection

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
df.info()
print("\n" + "=" * 60)
print("COLUMN DATA TYPES")
print("=" * 60)
print(df.dtypes)

# 2.3 Data Cleaning

# Missing values
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print("MISSING VALUES:")
print(missing_df[missing_df['Missing Count'] > 0] if missing.sum() > 0 else "No missing values found!")

# Duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates} duplicates. New shape: {df.shape}")

# Drop serial index column
if 'C' in df.columns:
    df = df.drop(columns=['C'])
    print("Dropped 'C' column (serial index).")

print(f"\nCleaned Dataset Shape: {df.shape}")
df.head()

# Fill missing values
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"Filled '{col}' with median: {df[col].median()}")

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"Filled '{col}' with mode: {df[col].mode()[0]}")

print(f"\nRemaining missing values: {df.isnull().sum().sum()}")

# ---
# 3. Exploratory Data Analysis (EDA)
# 3.1 Descriptive Statistics

print("NUMERICAL FEATURES")
print(df.describe().round(2))
print("\nCATEGORICAL FEATURES")
print(df.describe(include='object'))
print("\nTARGET DISTRIBUTION (Stages):")
stage_counts = df['Stages'].value_counts()
print(stage_counts)
print(f"\nPercentage:")
print((stage_counts / len(df) * 100).round(2))

# 3.2 Visual Analysis

age_map = {'18-34': 26, '35-50': 42.5, '51-64': 57.5, '65+': 70}
systolic_map = {'90 - 100': 95, '101 - 110': 105, '111 - 120': 115, '121 - 130': 125, '131 - 140': 135, '141 - 150': 145, '151 - 160': 155, '161 - 170': 165, '171 - 180': 175, '181 - 190': 185, '191 - 200': 195}
diastolic_map = {'51 - 60': 55.5, '61 - 70': 65.5, '70 - 80': 75, '81 - 90': 85, '91 - 100': 95, '101 - 110': 105, '111 - 120': 115, '121 - 130': 125, '131 - 140': 135, '141 - 150': 145}

df_plot = df.copy()
df_plot['Age_num'] = df_plot['Age'].map(age_map)
df_plot['Systolic_num'] = df_plot['Systolic'].map(systolic_map)
df_plot['Diastolic_num'] = df_plot['Diastolic'].map(diastolic_map)

# 1. Gender Distribution (Fixed missing column logic)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# Using 'Patient' as a placeholder since 'Gender' was missing in your original snippet
sns.countplot(data=df, x='Patient', ax=axs[0], palette='pastel')
axs[0].set_title('Distribution by Patient Category')

# Pie chart for proportions
df['Patient'].value_counts().plot.pie(autopct='%1.1f%%', ax=axs[1], colors=sns.color_palette('pastel'))
axs[1].set_title('Proportion Representation')
axs[1].set_ylabel('')

plt.tight_layout()
plt.savefig(os.path.join('plots','gender_distribution.png'), dpi=150, bbox_inches='tight')
# plt.show()  # suppressed for script mode

# 2. Hypertension Stages Distribution (FIXED OVERLAP)
fig, ax = plt.subplots(figsize=(10, 6)) # Increased width
sns.countplot(data=df, x='Stages', ax=ax, palette='viridis', edgecolor='black')

# SOLUTION: Rotate labels 45 degrees and align to right
plt.xticks(rotation=45, ha='right') 

ax.set_title('Distribution of Hypertension Stages', fontsize=14, fontweight='bold')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontweight='bold', xytext=(0, 5),
                textcoords='offset points')

plt.tight_layout()
plt.savefig(os.path.join('plots','stage_distribution.png'), dpi=150, bbox_inches='tight')
# plt.show()

# 3. Correlation between Systolic and Diastolic
corr = df_plot[['Systolic_num','Diastolic_num']].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0)
plt.title('Systolic vs Diastolic Correlation')
plt.tight_layout()
plt.savefig(os.path.join('plots','systolic_diastolic_corr.png'), dpi=150, bbox_inches='tight')
# plt.show()

# 4. TakeMedication vs. Severity (FIXED OVERLAP)
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, x='TakeMedication', hue='Stages', ax=ax, palette='viridis')
ax.set_title('Medication Status vs Hypertension Stages')

# SOLUTION: Rotate labels here too for consistency
plt.xticks(rotation=45, ha='right')
# Move legend so it doesn't cover bars
plt.legend(title='Stages', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join('plots','medication_severity.png'), dpi=150, bbox_inches='tight')
# plt.show()

# 5. Age Group vs Hypertension Stages (FIXED OVERLAP)
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(data=df, x='Age', hue='Stages', ax=ax, palette='Set2')
ax.set_title('Age Group vs Hypertension Stages')

plt.xticks(rotation=45, ha='right')
plt.legend(title='Stages', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join('plots','age_stage_count.png'), dpi=150, bbox_inches='tight')
# plt.show()

# 6. Pairplot: Systolic vs Diastolic across stages
# Note: Pairplot creates its own figure, so we don't use 'ax'
g = sns.pairplot(df_plot[['Systolic_num','Diastolic_num','Stages']], hue='Stages',
                 palette='viridis', diag_kind='kde',
                 plot_kws={'alpha':0.6, 'edgecolor':'black', 'linewidth':0.3})
g.fig.suptitle('Pairplot of Systolic and Diastolic by Stage', fontsize=16, fontweight='bold', y=1.02)
plt.savefig(os.path.join('plots','pairplot.png'), dpi=150, bbox_inches='tight')
# plt.show()

# ---
# 4. Feature Engineering & Selection



target = 'Stages'
features = [col for col in df.columns if col != target]
numerical_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df[features].select_dtypes(include=['object']).columns.tolist()

print("FEATURE SUMMARY")
print(f"Target: {target}")
print(f"Numerical ({len(numerical_features)}): {numerical_features}")
print(f"Categorical ({len(categorical_features)}): {categorical_features}")

# Encode target & build preprocessor
le_target = LabelEncoder()
y = le_target.fit_transform(df[target])
class_names = le_target.classes_
print(f"Classes: {list(class_names)}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ] if categorical_features else [
        ('num', StandardScaler(), numerical_features)
    ])

X = df[features]
print(f"Feature matrix: {X.shape}, Target: {y.shape}")

# Feature Importance (Random Forest)
X_processed = preprocessor.fit_transform(X)
if categorical_features:
    ohe = preprocessor.named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numerical_features + cat_feature_names
else:
    all_feature_names = numerical_features

rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X_processed, y)

importance_df = pd.DataFrame({
    'Feature': all_feature_names, 'Importance': rf_temp.feature_importances_
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(10, max(6, len(all_feature_names)*0.4)))
plt.barh(importance_df['Feature'], importance_df['Importance'],
         color=sns.color_palette('viridis', len(all_feature_names)))
plt.xlabel('Feature Importance')
plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nTop 5 Features:")
for _, row in importance_df.tail(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# ---
# 5. Data Splitting

# Merge rare classes to avoid stratification error
df['Stages'] = df['Stages'].replace({
    'HYPERTENSION (Stage-2).': 'HYPERTENSION (Stage-2)',
    'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS'
})

# Re-encode target after merging
le_target = LabelEncoder()
y = le_target.fit_transform(df[target])
class_names = le_target.classes_
print(f"Updated Classes: {list(class_names)}")
print(f"Updated Target Distribution:\n{df['Stages'].value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%")
print(f"Testing:  {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%")
print("\nTraining Target Distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  '{le_target.inverse_transform([u])[0]}': {c} ({c/len(y_train)*100:.1f}%")

# ---
# 6. Model Building
# 6.1 Training Multiple Algorithms

models = {
    'Logistic Regression': Pipeline([('pre', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
    'Decision Tree': Pipeline([('pre', preprocessor), ('clf', DecisionTreeClassifier(random_state=42))]),
    'Random Forest': Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))]),
    'Gradient Boosting': Pipeline([('pre', preprocessor), ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))]),
    'SVM': Pipeline([('pre', preprocessor), ('clf', SVC(kernel='rbf', probability=True, random_state=42))]),
    'KNN': Pipeline([('pre', preprocessor), ('clf', KNeighborsClassifier(n_neighbors=5))])
}

results = {}
print("=" * 70)
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("=" * 70)
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec,
                     'f1_score': f1, 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
                     'y_pred': y_pred}
    print(f"{name:<25} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")
print("=" * 70)

# ---
# 7. Performance Testing & Model Selection

comparison_df = pd.DataFrame({
    name: {'Accuracy': r['accuracy'], 'Precision': r['precision'],
           'Recall': r['recall'], 'F1-Score': r['f1_score'],
           'CV Mean': r['cv_mean'], 'CV Std': r['cv_std']}
    for name, r in results.items()
}).T.round(4).sort_values('Accuracy', ascending=False)
print("MODEL COMPARISON")
print(comparison_df.to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
comparison_df[['Accuracy','Precision','Recall','F1-Score']].plot(
    kind='bar', ax=axes[0], colormap='viridis', edgecolor='black', width=0.8)
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, 1.1)
axes[0].tick_params(axis='x', rotation=45)
cv_data = comparison_df[['CV Mean','CV Std']].sort_values('CV Mean', ascending=True)
axes[1].barh(cv_data.index, cv_data['CV Mean'], xerr=cv_data['CV Std'],
             color=sns.color_palette('viridis', len(cv_data)), edgecolor='black', capsize=5)
axes[1].set_title('Cross-Validation Accuracy (5-Fold)', fontsize=14, fontweight='bold')
axes[1].set_xlim(0, 1.1)
plt.tight_layout()
plt.savefig(os.path.join('plots','model_comparison.png'), dpi=150, bbox_inches='tight')
# plt.show()

# 7.2 Confusion Matrices
n_models = len(models)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(name, fontsize=12, fontweight='bold')
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join('plots','confusion_matrices.png'), dpi=150, bbox_inches='tight')
# plt.show()

# 7.3 Best Model Selection
best_model_name = max(results, key=lambda k: results[k]['f1_score'])
best_model = models[best_model_name]
best_results = results[best_model_name]

print("=" * 60)
print(f"BEST MODEL: {best_model_name}")
print("=" * 60)
print(f"  Accuracy:  {best_results['accuracy']:.4f}")
print(f"  Precision: {best_results['precision']:.4f}")
print(f"  Recall:    {best_results['recall']:.4f}")
print(f"  F1-Score:  {best_results['f1_score']:.4f}")
print(f"  CV:        {best_results['cv_mean']:.4f} +/- {best_results['cv_std']:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, best_results['y_pred'], target_names=class_names))

# ---
# 8. Model Deployment
# 8.1 Save the Best Model
joblib.dump(best_model, 'best_hypertension_model.pkl')
joblib.dump(le_target, 'label_encoder.pkl')

model_metadata = {
    'model_name': best_model_name, 'features': features,
    'numerical_features': numerical_features,
    'categorical_features': categorical_features,
    'class_names': list(class_names),
    'accuracy': float(best_results['accuracy']),
    'f1_score': float(best_results['f1_score']),
    'cv_accuracy': float(best_results['cv_mean'])
}
with open('model_metadata.json', 'w') as f:
    json_lib.dump(model_metadata, f, indent=2)

print("Model saved: best_hypertension_model.pkl")
print("Encoder saved: label_encoder.pkl")
print("Metadata saved: model_metadata.json")

# 8.2 Prediction Pipeline
def predict_hypertension(patient_data, model=None, label_encoder=None):
    if model is None:
        model = joblib.load('best_hypertension_model.pkl')
    if label_encoder is None:
        label_encoder = joblib.load('label_encoder.pkl')

    input_df = pd.DataFrame([patient_data])
    prediction = model.predict(input_df)
    predicted_stage = label_encoder.inverse_transform(prediction)[0]
    probabilities = model.predict_proba(input_df)[0]
    prob_dict = {label_encoder.inverse_transform([i])[0]: round(float(p), 4)
                 for i, p in enumerate(probabilities)}
    risk_score = round(float(max(probabilities) * 100), 1)

    return {'predicted_stage': predicted_stage, 'risk_score': risk_score, 'probabilities': prob_dict}

print("Prediction pipeline ready!")

# 8.3 Recommendation Module
def get_recommendations(prediction_result, patient_data):
    stage = prediction_result['predicted_stage']
    risk = prediction_result['risk_score']
    recs = {'stage': stage, 'risk_score': risk, 'recommendations': [], 'urgency': ''}

    stage_lower = str(stage).lower()
    if 'normal' in stage_lower or 'pre' in stage_lower:
        recs['urgency'] = 'LOW - Preventive Care'
        recs['recommendations'] = [
            "Maintain balanced diet low in sodium (< 2,300 mg/day)",
            "Exercise 150 min/week of moderate activity",
            "Monitor BP at home weekly",
            "Maintain healthy weight (BMI 18.5-24.9)",
            "Limit alcohol, avoid smoking",
            "Schedule annual health checkups"]
    elif 'stage-1' in stage_lower or 'stage 1' in stage_lower or stage_lower == '1':
        recs['urgency'] = 'MODERATE - Lifestyle Intervention'
        recs['recommendations'] = [
            "PRIORITY: Begin lifestyle modifications immediately",
            "Follow DASH diet",
            "Reduce sodium < 1,500 mg/day",
            "Exercise 30-40 min, 3-4x/week",
            "Monitor BP daily",
            "Consider stress management (yoga, meditation)",
            "Follow-up within 3-6 months",
            "Discuss medication with healthcare provider"]
    elif 'stage-2' in stage_lower or 'stage 2' in stage_lower or stage_lower == '2':
        recs['urgency'] = 'HIGH - Medical Intervention Required'
        recs['recommendations'] = [
            "URGENT: Consult physician immediately",
            "Medication likely required",
            "Strict dietary control, very low sodium",
            "Daily BP monitoring (morning and evening)",
            "Avoid high-stress activities",
            "Follow-ups every 1-3 months",
            "Consider cardiologist referral"]
    else:
        recs['urgency'] = 'CRITICAL - Emergency Attention'
        recs['recommendations'] = [
            "EMERGENCY: Seek immediate medical attention",
            "Do NOT delay treatment",
            "Follow all prescribed medications strictly",
            "Continuous BP monitoring required",
            "Lifestyle overhaul under supervision",
            "Weekly/bi-weekly follow-ups"]

    # protect against categorical ranges by mapping to numeric
    sys_val = patient_data.get('Systolic', 0)
    dia_val = patient_data.get('Diastolic', 0)
    # maps defined earlier in script
    try:
        sys_num = systolic_map.get(sys_val, float(sys_val))
    except Exception:
        sys_num = 0
    try:
        dia_num = diastolic_map.get(dia_val, float(dia_val))
    except Exception:
        dia_num = 0
    if sys_num > 180 or dia_num > 120:
        recs['recommendations'].insert(0, "HYPERTENSIVE CRISIS: Seek emergency care NOW!")
    return recs

def print_recommendations(recs):
    print("=" * 60)
    print("HYPERTENSION ASSESSMENT REPORT")
    print("=" * 60)
    print(f"Predicted Stage: {recs['stage']}")
    print(f"Risk Score: {recs['risk_score']}%")
    print(f"Urgency: {recs['urgency']}")
    print("\nRecommendations:")
    for i, r in enumerate(recs['recommendations'], 1):
        print(f"  {i}. {r}")
    print("=" * 60)
    print("Decision-support only. Consult a healthcare professional.")

print("Recommendation module ready!")

# 9. System Testing
print("=" * 60)
print("SCENARIO 1: Preventive Health Screening")
print("=" * 60)
sample = X_test.iloc[0].to_dict()
print(f"Patient: {sample}")
result = predict_hypertension(sample, model=best_model, label_encoder=le_target)
print(f"Prediction: {result['predicted_stage']}, Risk: {result['risk_score']}%")

print("\n" + "=" * 60)
print("SCENARIO 2: Hypertensive Monitoring")
print("=" * 60)
if len(X_test) > 1:
    sample2 = X_test.iloc[1].to_dict()
    result2 = predict_hypertension(sample2, model=best_model, label_encoder=le_target)
    print(f"Prediction: {result2['predicted_stage']}, Risk: {result2['risk_score']}%")
    print_recommendations(get_recommendations(result2, sample2))

print("\n" + "=" * 60)
print("SCENARIO 3: Emergency Triage")
print("=" * 60)
if len(X_test) > 2:
    sample3 = X_test.iloc[2].to_dict()
    result3 = predict_hypertension(sample3, model=best_model, label_encoder=le_target)
    print(f"Prediction: {result3['predicted_stage']}, Risk: {result3['risk_score']}%")
    print_recommendations(get_recommendations(result3, sample3))

# ---
# 10. Web Application (Flask)
# Generate a lightweight Flask application for deployment

flask_app_code = [
    "from flask import Flask, render_template, request, jsonify", 
    "import joblib, pandas as pd, json",
    "\napp = Flask(__name__)",
    "model = joblib.load('best_hypertension_model.pkl')",
    "encoder = joblib.load('label_encoder.pkl')",
    "with open('model_metadata.json') as f:",
    "    meta = json.load(f)",
    "\n@ app.route('/')",
    "def index():",
    "    return render_template('index.html', metadata=meta)",
    "\n@ app.route('/analyze', methods=['POST'])",
    "def analyze():",
    "    payload = request.get_json()",
    "    df_in = pd.DataFrame([payload])",
    "    pred = model.predict(df_in)[0]",
    "    stage = encoder.inverse_transform([pred])[0]",
    "    probs = model.predict_proba(df_in)[0]",
    "    prob_map = {encoder.inverse_transform([i])[0]: round(float(p),4) for i,p in enumerate(probs)}",
    "    score = round(max(probs)*100,1)",
    "    return jsonify({'prediction':stage,'score':score,'probs':prob_map})",
    "\nif __name__ == '__main__':",
    "    app.run(host='0.0.0.0', port=5000, debug=True)",
]

import os
os.makedirs('templates', exist_ok=True)
# copy template if exists
import shutil
if os.path.exists('flask_template.html'):
    shutil.copy('flask_template.html', 'templates/index.html')

with open('app.py', 'w') as f:
    f.write('\n'.join(flask_app_code))

print('Flask application generated and written to app.py')