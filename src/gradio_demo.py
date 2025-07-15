import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Mock data generation for demo (replace with your actual data loading)
def generate_mock_data():
    np.random.seed(42)
    n_samples = 4269
    
    # Generate synthetic data similar to your dataset
    data = {
        'no_of_dependents': np.random.randint(0, 6, n_samples),
        'education': np.random.choice([' Graduate', ' Not Graduate'], n_samples),
        'self_employed': np.random.choice([' Yes', ' No'], n_samples),
        'income_annum': np.random.normal(5000000, 2000000, n_samples),
        'loan_amount': np.random.normal(15000000, 8000000, n_samples),
        'loan_term': np.random.choice(range(2, 21), n_samples),
        'cibil_score': np.random.normal(600, 100, n_samples),
        'residential_assets_value': np.random.exponential(5000000, n_samples),
        'commercial_assets_value': np.random.exponential(3000000, n_samples),
        'luxury_assets_value': np.random.exponential(2000000, n_samples),
        'bank_asset_value': np.random.exponential(4000000, n_samples),
    }
    
    # Create loan_status based on cibil_score (main predictor from your analysis)
    loan_status = []
    for score in data['cibil_score']:
        if score > 550:
            loan_status.append(' Approved' if np.random.random() > 0.15 else ' Rejected')
        else:
            loan_status.append(' Rejected' if np.random.random() > 0.15 else ' Approved')
    
    data['loan_status'] = loan_status
    
    return pd.DataFrame(data)

# Load and prepare data
def prepare_model():
    # Generate mock data (replace with your actual data loading)
    df = generate_mock_data()
    
    # Create dummy variables
    loan_dummies = pd.get_dummies(df)
    loan_dummies.rename(columns={
        'education_ Graduate': 'education',
        'self_employed_ Yes': 'self_employed',
        'loan_status_ Approved': 'loan_status'
    }, inplace=True)
    
    # Drop redundant columns
    cols_to_drop = ['education_ Not Graduate', 'self_employed_ No', 'loan_status_ Rejected']
    loan_dummies = loan_dummies.drop([col for col in cols_to_drop if col in loan_dummies.columns], axis=1)
    
    # Separate features and target
    y = loan_dummies['loan_status']
    X = loan_dummies.drop(['loan_status'], axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=5,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    return rf_model, X.columns.tolist()

# Initialize model
model, feature_names = prepare_model()

def predict_loan_approval(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
):
    # Prepare input data
    input_data = {
        'no_of_dependents': no_of_dependents,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value,
        'education': 1 if education == "Graduate" else 0,
        'self_employed': 1 if self_employed == "Yes" else 0
    }
    
    # Create DataFrame with correct column order
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    # Get feature importance for this prediction
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Format result
    result = "✅ **APPROVED**" if prediction == 1 else "❌ **REJECTED**"
    confidence = f"Confidence: {max(probability):.2%}"
    
    # Format top features
    feature_text = "\n**Top 5 Important Features:**\n"
    for feature, importance in top_features:
        feature_text += f"• {feature}: {importance:.3f}\n"
    
    # Add interpretation based on your analysis
    interpretation = "\n**Key Insights:**\n"
    if cibil_score > 550:
        interpretation += "• Credit score is above the critical threshold (550) ✓\n"
    else:
        interpretation += "• Credit score is below the critical threshold (550) ⚠️\n"
    
    if loan_term <= 4:
        interpretation += "• Short loan term increases approval chances ✓\n"
    elif loan_term > 10:
        interpretation += "• Long loan term may reduce approval chances ⚠️\n"
    
    if income_annum > 5000000:
        interpretation += "• Above median annual income ✓\n"
    
    return f"{result}\n{confidence}\n{feature_text}{interpretation}"

# Create Gradio interface
with gr.Blocks(title="Loan Prediction System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏦 Loan Approval Prediction System
    
    This application predicts loan approval based on various financial and personal factors.
    The model achieves **97%+ accuracy** using Random Forest algorithm.
    
    ## Key Findings from Analysis:
    - **Credit Score (CIBIL)** is the most important factor
    - Scores above 550 significantly increase approval chances
    - Short-term loans (2-4 years) have higher approval rates
    - Higher annual income correlates with loan approval
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👤 Personal Information")
            no_of_dependents = gr.Slider(
                minimum=0, maximum=5, step=1, value=2,
                label="Number of Dependents"
            )
            education = gr.Radio(
                choices=["Graduate", "Not Graduate"],
                value="Graduate",
                label="Education Level"
            )
            self_employed = gr.Radio(
                choices=["Yes", "No"],
                value="No",
                label="Self Employed"
            )
            
            gr.Markdown("### 💰 Financial Information")
            income_annum = gr.Number(
                value=5000000,
                label="Annual Income (₹)",
                info="Enter your annual income in rupees"
            )
            loan_amount = gr.Number(
                value=15000000,
                label="Loan Amount (₹)",
                info="Enter requested loan amount in rupees"
            )
            loan_term = gr.Slider(
                minimum=2, maximum=20, step=1, value=4,
                label="Loan Term (Years)"
            )
            cibil_score = gr.Slider(
                minimum=300, maximum=850, step=1, value=650,
                label="CIBIL Score",
                info="Credit score (300-850)"
            )
        
        with gr.Column():
            gr.Markdown("### 🏠 Asset Information")
            residential_assets_value = gr.Number(
                value=5000000,
                label="Residential Assets Value (₹)",
                info="Value of residential properties"
            )
            commercial_assets_value = gr.Number(
                value=3000000,
                label="Commercial Assets Value (₹)",
                info="Value of commercial properties"
            )
            luxury_assets_value = gr.Number(
                value=2000000,
                label="Luxury Assets Value (₹)",
                info="Value of luxury items"
            )
            bank_asset_value = gr.Number(
                value=4000000,
                label="Bank Assets Value (₹)",
                info="Value of bank deposits/investments"
            )
            
            gr.Markdown("### 🔮 Prediction")
            predict_btn = gr.Button("Predict Loan Approval", variant="primary", size="lg")
            
            result_output = gr.Markdown(label="Prediction Result")
    
    # Examples
    gr.Markdown("### 📝 Try These Examples:")
    examples = gr.Examples(
        examples=[
            [2, "Graduate", "No", 6000000, 20000000, 4, 700, 8000000, 5000000, 3000000, 6000000],
            [1, "Graduate", "Yes", 8000000, 25000000, 2, 750, 10000000, 8000000, 5000000, 8000000],
            [3, "Not Graduate", "No", 3000000, 10000000, 10, 500, 2000000, 1000000, 500000, 2000000],
            [0, "Graduate", "No", 10000000, 30000000, 5, 800, 15000000, 12000000, 8000000, 10000000],
        ],
        inputs=[
            no_of_dependents, education, self_employed, income_annum, loan_amount,
            loan_term, cibil_score, residential_assets_value, commercial_assets_value,
            luxury_assets_value, bank_asset_value
        ]
    )
    
    # Connect button to prediction function
    predict_btn.click(
        fn=predict_loan_approval,
        inputs=[
            no_of_dependents, education, self_employed, income_annum, loan_amount,
            loan_term, cibil_score, residential_assets_value, commercial_assets_value,
            luxury_assets_value, bank_asset_value
        ],
        outputs=result_output
    )
    
    gr.Markdown("""
    ### 📊 Model Performance
    - **Accuracy**: 97.3%
    - **Precision**: 97.8%
    - **Recall**: 97.9%
    - **F1 Score**: 97.9%
    
    ### 🔍 About the Model
    This Random Forest model was trained on loan application data and uses the following key insights:
    - Credit score is the most important predictor
    - Loan term and annual income are significant factors
    - Asset values provide additional context
    - Demographic factors have minimal impact
    """)

if __name__ == "__main__":
    demo.launch()