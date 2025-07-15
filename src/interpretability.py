# Model Interpretability Analysis for Loan Prediction
# This script provides comprehensive interpretability analysis using SHAP and LIME

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

# Interpretability libraries
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance

# Install required packages (run in your environment):
# pip install shap lime

class LoanModelInterpreter:
    def __init__(self, model, X_train, X_test, y_train, y_test, feature_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        
    def shap_analysis(self):
        """Comprehensive SHAP analysis for global and local interpretability"""
        print("=== SHAP Analysis ===")
        
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)
        
        # For binary classification, we need the positive class SHAP values
        if len(shap_values) == 2:
            shap_values_positive = shap_values[1]  # Approved class
        else:
            shap_values_positive = shap_values
            
        # 1. Summary Plot - Shows feature importance and impact direction
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_positive, self.X_test, 
                         feature_names=self.feature_names, show=False)
        plt.title("SHAP Summary Plot - Feature Impact on Loan Approval")
        plt.tight_layout()
        plt.show()
        
        # 2. Feature Importance Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_positive, self.X_test, 
                         feature_names=self.feature_names, 
                         plot_type="bar", show=False)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.show()
        
        # 3. Waterfall plot for individual predictions
        print("\n--- Individual Prediction Explanations ---")
        
        # Explain a few individual predictions
        for i in [0, 1, 2]:
            if i < len(self.X_test):
                plt.figure(figsize=(10, 6))
                shap.waterfall_plot(
                    shap.Explanation(values=shap_values_positive[i], 
                                   base_values=explainer.expected_value[1] if len(shap_values) == 2 else explainer.expected_value,
                                   data=self.X_test.iloc[i],
                                   feature_names=self.feature_names),
                    show=False
                )
                plt.title(f"SHAP Waterfall Plot - Sample {i+1}")
                plt.tight_layout()
                plt.show()
        
        # 4. Dependence plots for key features
        key_features = ['cibil_score', 'income_annum', 'loan_amount']
        
        for feature in key_features:
            if feature in self.feature_names:
                plt.figure(figsize=(10, 6))
                feature_idx = list(self.feature_names).index(feature)
                shap.dependence_plot(feature_idx, shap_values_positive, 
                                   self.X_test, feature_names=self.feature_names,
                                   show=False)
                plt.title(f"SHAP Dependence Plot - {feature}")
                plt.tight_layout()
                plt.show()
        
        return shap_values_positive
    
    def lime_analysis(self):
        """LIME analysis for local interpretability"""
        print("\n=== LIME Analysis ===")
        
        # Initialize LIME explainer
        explainer = LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            class_names=['Rejected', 'Approved'],
            mode='classification'
        )
        
        # Explain a few individual predictions
        for i in [0, 1, 2]:
            if i < len(self.X_test):
                # Get explanation
                exp = explainer.explain_instance(
                    self.X_test.iloc[i].values,
                    self.model.predict_proba,
                    num_features=len(self.feature_names)
                )
                
                # Show in notebook
                print(f"\n--- LIME Explanation for Sample {i+1} ---")
                print(f"Prediction: {'Approved' if self.model.predict([self.X_test.iloc[i]])[0] == 1 else 'Rejected'}")
                print(f"Probability: {self.model.predict_proba([self.X_test.iloc[i]])[0]}")
                
                # Display explanation
                fig = exp.as_pyplot_figure()
                fig.suptitle(f'LIME Explanation - Sample {i+1}', fontsize=16)
                plt.tight_layout()
                plt.show()
    
    def permutation_importance_analysis(self):
        """Permutation importance analysis"""
        print("\n=== Permutation Importance Analysis ===")
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.model, self.X_test, self.y_test, 
            n_repeats=10, random_state=42
        )
        
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'], 
                xerr=importance_df['std'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Permutation Importance')
        plt.title('Permutation Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def fairness_analysis(self):
        """Analyze model fairness across different groups"""
        print("\n=== Fairness Analysis ===")
        
        # Create test predictions
        y_pred = self.model.predict(self.X_test)
        
        # Analyze performance across different groups
        test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        test_df['y_true'] = self.y_test.values
        test_df['y_pred'] = y_pred
        
        # Education fairness
        if 'education' in self.feature_names:
            education_groups = test_df.groupby('education').agg({
                'y_true': 'mean',
                'y_pred': 'mean'
            }).round(3)
            print("Performance by Education:")
            print(education_groups)
        
        # Self-employed fairness
        if 'self_employed' in self.feature_names:
            self_employed_groups = test_df.groupby('self_employed').agg({
                'y_true': 'mean',
                'y_pred': 'mean'
            }).round(3)
            print("\nPerformance by Self-Employment Status:")
            print(self_employed_groups)
        
        # Number of dependents fairness
        if 'no_of_dependents' in self.feature_names:
            # Create groups for analysis
            test_df['dependents_group'] = pd.cut(test_df['no_of_dependents'], 
                                               bins=[0, 1, 2, 5], 
                                               labels=['0-1', '2', '3+'])
            
            dependents_groups = test_df.groupby('dependents_group').agg({
                'y_true': 'mean',
                'y_pred': 'mean'
            }).round(3)
            print("\nPerformance by Number of Dependents:")
            print(dependents_groups)
    
    def feature_interaction_analysis(self, shap_values):
        """Analyze feature interactions"""
        print("\n=== Feature Interaction Analysis ===")
        
        # Calculate interaction values
        explainer = shap.TreeExplainer(self.model)
        interaction_values = explainer.shap_interaction_values(self.X_test.iloc[:100])  # Subset for performance
        
        # Plot interaction heatmap
        plt.figure(figsize=(12, 10))
        shap.summary_plot(interaction_values, self.X_test.iloc[:100], 
                         feature_names=self.feature_names, show=False)
        plt.title("SHAP Interaction Values")
        plt.tight_layout()
        plt.show()
    
    def ethical_considerations_report(self):
        """Generate ethical considerations report"""
        print("\n=== Ethical Considerations Report ===")
        
        report = """
        ETHICAL CONSIDERATIONS FOR LOAN PREDICTION MODEL:
        
        1. FAIRNESS ASSESSMENT:
           - Ensure equal treatment across demographic groups
           - Monitor for disparate impact on protected classes
           - Regular auditing of approval rates by group
        
        2. TRANSPARENCY:
           - Model decisions should be explainable to applicants
           - Clear documentation of factors affecting loan approval
           - Regular model interpretability reviews
        
        3. BIAS MITIGATION:
           - Remove or reduce reliance on potentially biased features
           - Implement fairness constraints during training
           - Regular bias testing and monitoring
        
        4. REGULATORY COMPLIANCE:
           - Ensure compliance with fair lending laws
           - Document model development and validation process
           - Regular compliance audits
        
        5. CONTINUOUS MONITORING:
           - Track model performance over time
           - Monitor for drift in feature importance
           - Regular retraining with diverse data
        
        RECOMMENDATIONS:
        - Focus on financial capacity indicators (income, assets, credit score)
        - Minimize reliance on demographic characteristics
        - Implement regular fairness audits
        - Provide clear explanations for loan decisions
        - Establish appeals process for rejected applications
        """
        
        print(report)

# Example usage with your loan dataset
def run_interpretability_analysis():
    """
    Run this function with your trained model and data
    """
    
    # Example of how to use (adapt to your actual data):
    """
    # Assuming you have your trained model and data ready:
    
    # Initialize interpreter
    interpreter = LoanModelInterpreter(
        model=rf_opt,  # Your trained Random Forest model
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X_train.columns)
    )
    
    # Run all analyses
    shap_values = interpreter.shap_analysis()
    interpreter.lime_analysis()
    importance_df = interpreter.permutation_importance_analysis()
    interpreter.fairness_analysis()
    interpreter.feature_interaction_analysis(shap_values)
    interpreter.ethical_considerations_report()
    """
    
    print("Please initialize the LoanModelInterpreter with your trained model and data.")
    print("See the commented code above for usage example.")

# Additional utility functions for model validation
def validate_model_decisions(model, X_test, y_test, feature_names):
    """Validate that model decisions are based on appropriate features"""
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("=== Model Decision Validation ===")
    print("\nTop 5 Most Important Features:")
    print(importance_df.head())
    
    # Check if credit score is the most important (as expected)
    if 'cibil_score' in importance_df.iloc[0]['feature']:
        print("\n✓ GOOD: Credit score is the most important feature")
    else:
        print("\n⚠ WARNING: Credit score is not the most important feature")
    
    # Check if demographic features have low importance
    demographic_features = ['education', 'self_employed', 'no_of_dependents']
    demo_importance = importance_df[importance_df['feature'].isin(demographic_features)]
    
    if demo_importance['importance'].max() < 0.1:
        print("✓ GOOD: Demographic features have low importance")
    else:
        print("⚠ WARNING: Some demographic features have high importance")
        print(demo_importance)

def generate_model_card():
    """Generate a model card for documentation"""
    
    model_card = """
    ==========================================
    LOAN PREDICTION MODEL CARD
    ==========================================
    
    MODEL DETAILS:
    - Model Type: Random Forest Classifier
    - Purpose: Predict loan approval likelihood
    - Performance: ~97% accuracy on test set
    
    INTENDED USE:
    - Assist in loan approval decisions
    - Provide consistent evaluation criteria
    - Support fair lending practices
    
    FACTORS CONSIDERED:
    - Credit score (primary factor)
    - Income and financial capacity
    - Loan amount and terms
    - Asset values
    
    ETHICAL CONSIDERATIONS:
    - Regular fairness audits required
    - Human oversight for final decisions
    - Appeals process for rejected applications
    
    LIMITATIONS:
    - Model may not capture all relevant factors
    - Requires regular retraining and monitoring
    - Should not be used as sole decision-making tool
    
    MONITORING:
    - Performance tracking over time
    - Bias detection and mitigation
    - Regular model interpretability reviews
    """
    
    return model_card

if __name__ == "__main__":
    run_interpretability_analysis()
    print("\n" + generate_model_card())