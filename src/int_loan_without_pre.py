# # Direct Integration with Your Loan Prediction Code


# import shap
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.inspection import permutation_importance
# import warnings
# warnings.filterwarnings("ignore")

# # ============================================================================
# # ADD THIS SECTION AFTER YOUR EXISTING MODEL TRAINING
# # ============================================================================

# # Assuming your existing variables are:
# # rf_opt = your trained Random Forest model
# # X_train, X_test, y_train, y_test = your data splits
# # loan_dummies = your processed dataframe

# print("🔍 STARTING MODEL INTERPRETABILITY ANALYSIS")
# print("=" * 60)

# # 1. SHAP Analysis for Global Interpretability
# print("\n1. SHAP ANALYSIS - Understanding Feature Importance")
# print("-" * 50)

# # Initialize SHAP explainer
# explainer = shap.TreeExplainer(rf_opt)

# # Use subset for performance (adjust size based on your computational resources)
# X_shap = X_test.iloc[:200]  # Use first 200 samples
# shap_values = explainer.shap_values(X_shap)

# # Handle binary classification SHAP values
# if isinstance(shap_values, list) and len(shap_values) == 2:
#     shap_values_approved = shap_values[1]  # Values for "Approved" class
#     expected_value = explainer.expected_value[1]
# else:
#     shap_values_approved = shap_values
#     expected_value = explainer.expected_value

# # Feature importance plot
# plt.figure(figsize=(12, 8))
# shap.summary_plot(shap_values_approved, X_shap, 
#                  feature_names=X_shap.columns,
#                  plot_type="bar",
#                  show=False)
# plt.title("Feature Importance (SHAP)")
# plt.xlabel("SHAP Value")
# plt.ylabel("Feature")
# plt.tight_layout()
# plt.show()

# # 2. LIME Analysis for Local Interpretability   
# # 2. LIME Analysis for Local Interpretability
# print("\n2. LIME ANALYSIS - Understanding Local Decisions")
# print("-" * 50)

# # Initialize LIME explainer
# explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, 
#                                                    mode='classification',
#                                                    feature_names=X_train.columns,
#                                                    class_names=['Rejected', 'Approved'])

# # Explain individual decisions
# sample_idx = 0  # Change this to explore different samples
# exp = explainer.explain_instance(X_test.values[sample_idx], rf_opt.predict_proba, num_features=5)
# exp.show_in_notebook(show_table=True)
