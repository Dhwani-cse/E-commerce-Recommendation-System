import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("products.csv")

# Create combined feature
df['features'] = df['category'] + " " + df['description']

# Convert text to vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['features'])

# Compute similarity
similarity = cosine_similarity(tfidf_matrix)

# Recommendation function
def recommend(product_name):
    index = df[df['product_name'] == product_name].index[0]
    scores = list(enumerate(similarity[index]))
    
    # Sort by similarity
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    
    # Skip first (same product)
    for i in sorted_scores[1:6]:
        product = df.iloc[i[0]]['product_name']
        score = i[1]
        recommendations.append((product, score))
    
    return recommendations


# ---------------- STREAMLIT UI ---------------- #

st.title("🛍️ E-Commerce Recommendation System")
st.markdown("### Smart Product Suggestions using Machine Learning 🤖")
st.markdown("---")
st.markdown("💡 Recommends products using TF-IDF + Cosine Similarity")

# Category filter
category = st.selectbox("Select Category", df['category'].unique())

filtered_df = df[df['category'] == category]

# Product selection
product = st.selectbox("Choose a product", filtered_df['product_name'].values)

# Button
if st.button("Recommend"):
    results = recommend(product)

    st.subheader("Recommended Products:")
    
    for item, score in results:
        st.write(f"👉 {item} (Similarity: {round(score*100, 2)}%)")