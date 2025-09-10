import streamlit as st
import pandas as pd
import joblib
import re
import matplotlib.pyplot as plt

# --- Preprocess function ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

# --- Load saved model and TF-IDF vectorizer ---
model = joblib.load('sentiment_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

# --- Emoji mapping ---
def sentiment_with_emoji(label):
    if label == "Positive":
        return "😊 Positive"
    else:
        return "😞 Negative"

# --- Streamlit Interface ---
st.title("📊 Sentiment Analysis on Product Reviews")

st.markdown("### ✅ The uploaded Test CSV file should have the following columns:")
st.markdown("- **ProductId** : Unique identifier for the product")
st.markdown("- **UserId** : Unique identifier for the user")
st.markdown("- **Text** : Review text of the product")

uploaded_file = st.file_uploader("Upload Test CSV file (with above columns)", type=["csv"])

if uploaded_file is not None:
    new_reviews_df = pd.read_csv(uploaded_file)

    review_column = "Text"
    product_id_col = "ProductId"
    user_id_col = "UserId"

    # Preprocess, vectorize, predict
    new_reviews_df["Cleaned_Review"] = new_reviews_df[review_column].apply(preprocess_text)
    vectorized_reviews = tfidf.transform(new_reviews_df["Cleaned_Review"]).toarray()
    predicted_sentiments = model.predict(vectorized_reviews)
    
    new_reviews_df["Predicted_Sentiment"] = predicted_sentiments
    new_reviews_df["Sentiment_Label"] = new_reviews_df["Predicted_Sentiment"].map({1: "Positive", 0: "Negative"})
    new_reviews_df["Sentiment_With_Emoji"] = new_reviews_df["Sentiment_Label"].apply(sentiment_with_emoji)

    output_df = new_reviews_df[[product_id_col, user_id_col, "Predicted_Sentiment", "Sentiment_Label", "Sentiment_With_Emoji"]]

    # Display result table
    st.dataframe(output_df)

    # Product-wise pie charts in 2 per row
    st.markdown("### 📊 Sentiment Distribution Per Product")

    product_list = output_df[product_id_col].unique()

    # Iterate products and arrange charts in pairs
    for i in range(0, len(product_list), 2):
        cols = st.columns(2)

        for j in range(2):
            if i + j < len(product_list):
                product = product_list[i + j]
                product_data = output_df[output_df[product_id_col] == product]
                sentiment_counts = product_data["Sentiment_Label"].value_counts()

                fig, ax = plt.subplots(figsize=(3, 3))
                ax.pie(
                    sentiment_counts,
                    labels=sentiment_counts.index.map(lambda x: f"{x} ({sentiment_counts[x]})"),
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['#4CAF50', '#F44336']
                )
                ax.axis('equal')
                cols[j].markdown(f"#### 📦 Product ID: {product}")
                cols[j].pyplot(fig)

    # Download button
    csv = output_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name='Predicted_Results.csv',
        mime='text/csv'
    )
