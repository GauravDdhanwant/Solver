import streamlit as st
import pandas as pd
import openai
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Function to analyze data with OpenAI GPT
def analyze_data_with_llm(data, openai_api_key):
    openai.api_key = openai_api_key
    prompt = f"Analyze the following data and suggest machine learning problems that can be solved:\n\n{data.head().to_csv()}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert data scientist."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        raise ValueError(f"Error while querying OpenAI API: {e}")

def main():
    st.title("AI-Powered ML Problem Solver")
    
    # Input for OpenAI API key
    st.sidebar.header("OpenAI API Key")
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
    
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key to proceed.")
        return
    
    # Upload CSV file
    st.sidebar.header("Upload CSV Data")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.write(data.head())
        
        # Analyze data and identify problems
        with st.spinner("Analyzing data..."):
            try:
                suggestions = analyze_data_with_llm(data, openai_api_key)
                st.write("### Suggested Problems")
                st.write(suggestions)
            except Exception as e:
                st.error(f"Error: {e}")
                return
        
        # Ask user-specific questions
        problem_type = st.selectbox("Select the problem type", ["Regression", "Classification", "Clustering"])
        target_column = st.text_input("Enter the target column for prediction (if applicable):")
        
        # Proceed with ML pipeline if inputs are valid
        if st.button("Solve the Problem"):
            if target_column:
                st.write("### Solving the problem...")
                # Split data
                X = data.drop(columns=[target_column])
                y = data[target_column]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train a sample model
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                # Display results
                st.write("### Classification Report")
                st.text(classification_report(y_test, predictions))
                
                # Plot feature importance
                st.write("### Feature Importance")
                importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values(by="Importance", ascending=False)
                sns.barplot(data=importance, x="Importance", y="Feature")
                plt.title("Feature Importance")
                st.pyplot(plt.gcf())
                
                # Initialize session state for inputs
                if "user_inputs" not in st.session_state:
                    st.session_state.user_inputs = {col: "" for col in X.columns}
                
                # Allow user to test the model
                st.write("### Test the Model")
                for col in X.columns:
                    st.session_state.user_inputs[col] = st.text_input(
                        f"Enter value for {col}:", 
                        value=st.session_state.user_inputs.get(col, "")
                    )
                
                if st.button("Predict"):
                    # Ensure all inputs are provided
                    if all(st.session_state.user_inputs.values()):
                        input_data = {col: float(val) for col, val in st.session_state.user_inputs.items()}
                        input_df = pd.DataFrame([input_data])
                        prediction = model.predict(input_df)
                        st.success(f"Prediction: {prediction[0]}")
                    else:
                        st.error("Please provide inputs for all features.")
            else:
                st.error("Please specify the target column.")

if __name__ == "__main__":
    main()
