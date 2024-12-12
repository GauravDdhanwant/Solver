import streamlit as st
import pandas as pd
from openai import Completion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# OpenAI API Setup
openai_api_key = "YOUR_OPENAI_API_KEY"

def analyze_data_with_llm(data, openai_api_key):
    # Leverage GPT to analyze the dataset and suggest ML problems
    prompt = f"Analyze the following data and suggest machine learning problems that can be solved:\n\n{data.head().to_csv()}"
    response = Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        api_key=openai_api_key
    )
    return response.choices[0].text.strip()

def main():
    st.title("AI-Powered ML Problem Solver")
    
    # Upload CSV file
    st.sidebar.header("Upload CSV Data")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.write(data.head())
        
        # Analyze data and identify problems
        with st.spinner("Analyzing data..."):
            suggestions = analyze_data_with_llm(data, openai_api_key)
        
        st.write("### Suggested Problems")
        st.write(suggestions)
        
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
            else:
                st.error("Please specify the target column.")

if __name__ == "__main__":
    main()
