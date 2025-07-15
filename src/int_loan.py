# Practical Implementation Guide for Loan Model Interpretability
# This script shows how to integrate interpretability with your existing loan prediction code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# Step 1: Enhanced Model Training with Interpretability Focus
def train_interpretable_model(X_train, y_train, X_val, y_val):
    """Train model with interpretability considerations"""
    
    # Use fewer trees for better interpretability while maintaining performance
    rf_interpretable = RandomForestClassifier(
        n_estimators=100,  # Reduced for interpretability
        max_depth=10,      # Limited depth for interpretability
        min_samples_leaf=5, # Prevent overfitting
        random_state=42
    )
    
    rf_interpretable.fit(X_train, y_train)
    
    # Validate performance
    y_pred = rf_interpretable.predict(X_val)
    print("Model Performance:")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.3f}")
    print(f"Precision: {precision_score(y_val, y_pred):.3f}")
    print(f"Recall: {recall_score(y_val, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_val, y_pred):.3f}")
    
    return rf_interpretable

# Step 2: Quick SHAP Analysis Function
def quick_shap_analysis(model, X_test, feature_names, max_samples=100):
    """Quick SHAP analysis for immediate insights"""
    
    try:
        import shap
        
        # Use subset for performance
        X_subset = X_test.iloc[:max_samples]
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_subset)
        
        # Handle binary classification
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_plot = shap_values[1]  # Positive class
        else:
            shap_values_plot = shap_values
        
        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_plot, X_subset, 
                         feature_names=feature_names, 
                         plot_type="bar", show=False)
        plt.title("Feature Importance (SHAP)")
        plt.tight_layout()
        plt.show()
        
        # Detailed summary
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_plot, X_subset, 
                         feature_names=feature_names, show=False)
        plt.title("Feature Impact Analysis (SHAP)")
        plt.tight_layout()
        plt.show()
        
        return shap_values_plot
        
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return None

# Step 3: Fairness Assessment
def assess_fairness(model, X_test, y_test, feature_names):
    """Assess model fairness across different groups"""
    
    y_pred = model.predict(X_test)
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['actual'] = y_test
    test_df['predicted'] = y_pred
    
    print("=== FAIRNESS ASSESSMENT ===")
    
    # Overall statistics
    overall_approval_rate = test_df['predicted'].mean()
    print(f"Overall Approval Rate: {overall_approval_rate:.3f}")
    
    # Education fairness (if available)
    if 'education' in feature_names:
        education_stats = test_df.groupby('education').agg({
            'predicted': 'mean',
            'actual': 'mean'
        }).round(3)
        print("\nApproval Rates by Education:")
        print(education_stats)
        
        # Check for significant disparities
        ed_diff = education_stats['predicted'].max() - education_stats['predicted'].min()
        if ed_diff > 0.1:  # 10% difference threshold
            print(f"⚠ WARNING: Education disparity detected: {ed_diff:.3f}")
        else:
            print("✓ Education fairness acceptable")
    
    # Self-employment fairness (if available)
    if 'self_employed' in feature_names:
        self_emp_stats = test_df.groupby('self_employed').agg({
            'predicted': 'mean',
            'actual': 'mean'
        }).round(3)
        print("\nApproval Rates by Self-Employment:")
        print(self_emp_stats)
        
        # Check for disparities
        se_diff = self_emp_stats['predicted'].max() - self_emp_stats['predicted'].min()
        if se_diff > 0.1:
            print(f"⚠ WARNING: Self-employment disparity detected: {se_diff:.3f}")
        else:
            print("✓ Self-employment fairness acceptable")

# Step 4: Decision Explanation for Individual Cases
def explain_individual_decision(model, X_test, feature_names, sample_idx=0):
    """Explain individual loan decisions"""
    
    sample = X_test.iloc[sample_idx]
    prediction = model.predict([sample])[0]
    probability = model.predict_proba([sample])[0]
    
    print(f"=== INDIVIDUAL DECISION EXPLANATION ===")
    print(f"Sample Index: {sample_idx}")
    print(f"Prediction: {'APPROVED' if prediction == 1 else 'REJECTED'}")
    print(f"Probability: {probability[1]:.3f} (Approval)")
    
    # Feature contributions (simplified)
    feature_importance = model.feature_importances_
    feature_values = sample.values
    
    # Create explanation dataframe
    explanation_df = pd.DataFrame({
        'feature': feature_names,
        'value': feature_values,
        'importance': feature_importance,
        'contribution': feature_values * feature_importance
    }).sort_values('contribution', ascending=False)
    
    print("\nTop Contributing Factors:")
    print(explanation_df.head(5)[['feature', 'value', 'contribution']])
    
    # Visual explanation
    plt.figure(figsize=(10, 6))
    top_features = explanation_df.head(8)
    plt.barh(range(len(top_features)), top_features['contribution'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Contribution to Decision')
    plt.title(f'Decision Explanation - Sample {sample_idx}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Step 5: Model Validation Checklist
def validate_model_ethics(model, X_test, y_test, feature_names):
    """Comprehensive model validation checklist"""
    
    print("=== MODEL ETHICS VALIDATION ===")
    
    # 1. Feature importance check
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("1. Feature Importance Check:")
    print(importance_df.head())
    
    # Check if financial factors dominate
    financial_features = ['cibil_score', 'income_annum', 'loan_amount', 
                         'residential_assets_value', 'commercial_assets_value',
                         'luxury_assets_value', 'bank_asset_value']
    
    financial_importance = importance_df[
        importance_df['feature'].isin(financial_features)
    ]['importance'].sum()
    
    if financial_importance > 0.7:
        print("✓ GOOD: Financial factors dominate decision (>70%)")
    else:
        print("⚠ WARNING: Non-financial factors have high influence")
    
    # 2. Demographic feature check
    demographic_features = ['education', 'self_employed', 'no_of_dependents']
    demo_importance = importance_df[
        importance_df['feature'].isin(demographic_features)
    ]['importance'].sum()
    
    if demo_importance < 0.2:
        print("✓ GOOD: Demographic features have low importance (<20%)")
    else:
        print("⚠ WARNING: Demographic features have high importance")
    
    # 3. Performance consistency
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    if accuracy > 0.85 and precision > 0.8 and recall > 0.8:
        print("✓ GOOD: Model performance is acceptable")
    else:
        print("⚠ WARNING: Model performance may be inadequate")
    
    # 4. Decision consistency
    # Check if similar profiles get similar decisions
    print("\n4. Decision Consistency Check:")
    print("(Manual review recommended for similar profiles)")
    
    return {
        'financial_importance': financial_importance,
        'demographic_importance': demo_importance,
        'performance_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    }

# Step 6: Generate Interpretability Report
def generate_interpretability_report(model, X_test, y_test, feature_names):
    """Generate comprehensive interpretability report"""
    
    report = {
        'model_type': type(model).__name__,
        'feature_count': len(feature_names),
        'test_samples': len(X_test),
        'performance': {},
        'feature_importance': {},
        'ethical_assessment': {}
    }
    
    # Performance metrics
    y_pred = model.predict(X_test)
    report['performance'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    report['feature_importance'] = {
        'top_features': importance_df.head(5).to_dict('records'),
        'most_important': importance_df.iloc[0]['feature'],
        'least_important': importance_df.iloc[-1]['feature']
    }
    
    # Ethical assessment
    validation_results = validate_model_ethics(model, X_test, y_test, feature_names)
    report['ethical_assessment'] = validation_results
    
    return report

# Step 7: Complete Implementation Example
def complete_interpretability_workflow():
    """Complete workflow for interpretable loan prediction"""
    
    print("=== INTERPRETABLE LOAN PREDICTION WORKFLOW ===")
    print("This is an example workflow. Adapt to your actual data.")
    
    # Example data structure (replace with your actual data)
    # 1. Load and prepare your data
    loan_data = pd.read_csv('/Users/ahmedmostafa/Downloads/coding_loan_system/assets/loan_dummies.csv')
    X = loan_data.drop(['loan_status'], axis=1)
    y = loan_data['loan_status']
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Train interpretable model
    model = train_interpretable_model(X_train, y_train, X_test, y_test)
    
    # 4. Run interpretability analyses
    shap_values = quick_shap_analysis(model, X_test, X.columns)
    assess_fairness(model, X_test, y_test, X.columns)
    
    # 5. Explain individual decisions
    explain_individual_decision(model, X_test, X.columns, sample_idx=0)
    explain_individual_decision(model, X_test, X.columns, sample_idx=1)
    
    # 6. Validate ethics
    validation_results = validate_model_ethics(model, X_test, y_test, X.columns)
    
    # 7. Generate report
    report = generate_interpretability_report(model, X_test, y_test, X.columns)
    
    print("\\n=== INTERPRETABILITY REPORT ===")
    print(f"Model Type: {report['model_type']}")
    print(f"Test Accuracy: {report['performance']['accuracy']:.3f}")
    print(f"Most Important Feature: {report['feature_importance']['most_important']}")
    print(f"Financial Feature Importance: {report['ethical_assessment']['financial_importance']:.3f}")
    
    
    # print("\nPlease integrate this code with your actual loan prediction model.")
    # print("Key steps:")
    # print("1. Replace example data with your actual dataset")
    # print("2. Install required packages: pip install shap lime")
    # print("3. Run each analysis function with your model")
    # print("4. Review results for ethical considerations")
    # print("5. Document findings and implement monitoring")

if __name__ == "__main__":
    complete_interpretability_workflow()