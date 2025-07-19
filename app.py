import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("best_model.pkl")

# Define the exact feature order used during training
MANUAL_FEATURE_ORDER = [
    'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
]

# Define minimum required columns (must include either education or educational-num)
MIN_REQUIRED_COLS = [
    'age', 'workclass', 'fnlwgt', 'marital-status',
    'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week',
    'native-country'
]

# Define categorical encoders with complete class lists
education_encoder = LabelEncoder()
education_encoder.classes_ = np.array([
    'Bachelors', 'Masters', 'PhD', 'HS-grad', 
    'Assoc', 'Some-college', 'Other', 'Unknown'
])

workclass_encoder = LabelEncoder()
workclass_encoder.classes_ = np.array(['Private', 'Self-emp', 'Govt', 'Other', 'Unknown'])

occupation_encoder = LabelEncoder()
occupation_encoder.classes_ = np.array([
    'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
    'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
    'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
    'Transport-moving', 'Priv-house-serv', 'Protective-serv', 
    'Armed-Forces', 'Unknown'
])

# Set up Streamlit app
st.set_page_config(page_title="Salary Classifier", layout="centered")
st.title("💰 Employee Salary Classification")
st.markdown("Predict >50K or ≤50K based on employee details")

# =============================================
# Individual Prediction
# =============================================
with st.expander("🔍 Individual Prediction", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 90, 30)
        workclass = st.selectbox("Workclass", workclass_encoder.classes_[:-1])
        fnlwgt = st.number_input("Final Weight", 10000, 1000000, 200000)
        education = st.selectbox("Education", education_encoder.classes_[:-1])
        education_num = st.slider("Education Num", 1, 16, 9)
        marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
        
    with col2:
        occupation = st.selectbox("Occupation", occupation_encoder.classes_[:-1])
        relationship = st.selectbox("Relationship", ["Husband", "Wife", "Other"])
        race = st.selectbox("Race", ["White", "Black", "Other"])
        sex = st.selectbox("Sex", ["Male", "Female"])
        capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
        capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
        hours_week = st.slider("Hours/Week", 1, 80, 40)
        country = st.selectbox("Country", ["US", "Other"])

    # Encode categorical features
    input_data = {
        'age': age,
        'workclass': workclass_encoder.transform([workclass])[0],
        'fnlwgt': fnlwgt,
        'education': education_encoder.transform([education])[0],
        'educational-num': education_num,
        'marital-status': 1 if marital_status == "Married" else 0,
        'occupation': occupation_encoder.transform([occupation])[0],
        'relationship': 1 if relationship == "Husband" else 0,
        'race': 1 if race == "White" else 0,
        'sex': 1 if sex == "Male" else 0,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_week,
        'native-country': 1 if country == "US" else 0
    }

    if st.button("Predict Salary"):
        input_df = pd.DataFrame([input_data])[MANUAL_FEATURE_ORDER]
        try:
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            st.success(f"Prediction: {'>50K' if pred == 1 else '<=50K'}")
            st.write(f"Confidence: {max(proba)*100:.1f}%")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# =============================================
# Batch Prediction
# =============================================
with st.expander("📂 Batch Prediction", expanded=True):
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        try:
            batch_data = pd.read_csv(uploaded_file)
            
            # Check for required columns
            missing = [col for col in MIN_REQUIRED_COLS if col not in batch_data.columns]
            
            # Handle education data - must have either education or educational-num
            has_education = 'education' in batch_data.columns
            has_ednum = 'educational-num' in batch_data.columns
            
            if not has_education and not has_ednum:
                st.error("❌ Must include either 'education' or 'educational-num' column")
                st.stop()
                
            # Create education column if only educational-num exists
            if has_ednum and not has_education:
                education_mapping = {
                    1: 'HS-grad', 2: 'Some-college', 3: 'Assoc',
                    4: 'Bachelors', 5: 'Masters', 6: 'PhD',
                    7: 'Prof-school', 8: 'Doctorate', 9: 'HS-grad',
                    10: 'Some-college', 11: 'Assoc', 12: 'Bachelors',
                    13: 'Masters', 14: 'PhD', 15: 'Prof-school',
                    16: 'Doctorate'
                }
                batch_data['education'] = batch_data['educational-num'].map(education_mapping).fillna('Unknown')
                # st.warning("Converted 'educational-num' to 'education' labels")
            
            if missing:
                st.error(f"❌ Missing required columns: {', '.join(missing)}")
                st.stop()
            
            # Safe encoding function
            def safe_encode(series, encoder):
                try:
                    return encoder.transform(series.astype(str))
                except ValueError:
                    known_labels = set(encoder.classes_)
                    return series.apply(lambda x: encoder.transform(['Unknown'])[0] if str(x) not in known_labels else encoder.transform([str(x)])[0])
            
            # Convert categorical columns
            if 'education' in batch_data.columns:
                batch_data['education'] = safe_encode(batch_data['education'], education_encoder)
            if 'workclass' in batch_data.columns:
                batch_data['workclass'] = safe_encode(batch_data['workclass'], workclass_encoder)
            if 'occupation' in batch_data.columns:
                batch_data['occupation'] = safe_encode(batch_data['occupation'], occupation_encoder)
            
            # Convert other categoricals
            categorical_mappings = {
                'marital-status': {'Married': 1, 'Single': 0, 'Divorced': 0},
                'relationship': {'Husband': 1, 'Wife': 0, 'Other': 0},
                'race': {'White': 1, 'Black': 0, 'Other': 0},
                'sex': {'Male': 1, 'Female': 0},
                'native-country': {'US': 1, 'Other': 0}
            }
            
            for col, mapping in categorical_mappings.items():
                if col in batch_data.columns:
                    batch_data[col] = batch_data[col].map(mapping).fillna(0)
            
            # Ensure we have all expected columns
            batch_data = batch_data[MANUAL_FEATURE_ORDER]
            
            st.write("Processed data preview:")
            st.dataframe(batch_data.head())
            
            if st.button("Predict Batch"):
                try:
                    predictions = model.predict(batch_data)
                    probas = model.predict_proba(batch_data)
                    
                    results = batch_data.copy()
                    results['Prediction'] = ['>50K' if p == 1 else '<=50K' for p in predictions]
                    results['Confidence'] = np.max(probas, axis=1)
                    
                    st.success(f"✅ Predicted {len(results)} records")
                    st.dataframe(results.head())
                    
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results",
                        csv,
                        "predictions.csv",
                        "text/csv"
                    )
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    
        except Exception as e:
            st.error(f"File processing error: {str(e)}")

# =============================================
# Sample Data Format
# =============================================
with st.expander("💡 Sample Data Format"):
    st.write("""
    **Required columns (must include either education or educational-num):**  
    age, workclass, fnlwgt, [education or educational-num],  
    marital-status, occupation, relationship, race, sex,  
    capital-gain, capital-loss, hours-per-week, native-country  
    
    **Education Options:**  
    Bachelors, Masters, PhD, HS-grad, Assoc, Some-college, Other  
    
    **Educational-Num Mapping:**  
    1-9: HS-grad to Assoc  
    10-16: Bachelors to Doctorate  
    """)
    
    st.write("Example CSV with education text:")
    st.code("""age,workclass,fnlwgt,education,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country
32,Private,125000,Bachelors,Married,Exec-managerial,Husband,White,Male,0,0,40,US""")

    st.write("Example CSV with educational-num:")
    st.code("""age,workclass,fnlwgt,educational-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country
32,Private,125000,13,Married,Exec-managerial,Husband,White,Male,0,0,40,US""") 