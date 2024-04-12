import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Load the dataset
data = pd.read_csv('cleaned_books_df.csv')

# Preprocess the data
data['Features'] = data['Author'] + ' ' + data['Main Genre'] + ' ' + data['Sub Genre']
data['Features'] = data['Features'].fillna('')

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the TF-IDF Vectorizer on the 'Features' column
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Features'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(titles):
    recommendations = []
    for title in titles:
    	# Check if the title exists in the dataset
        if title not in data['Title'].values:
            st.warning(f'The book "{title}" is not in the database.')
            return None

        # Get the index of the book that matches the title
        idx = data[data['Title'] == title].index[0]

        # Get the pairwise similarity scores of all books with that book
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the books based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the top 10 most similar books
        sim_scores = sim_scores[1:11]  # Change to 10 recommendations

        # Get the book indices
        book_indices = [i[0] for i in sim_scores]

        # Get the top 10 most similar books and their scores
        recommended_books = data[['Title', 'Author', 'Main Genre', 'Sub Genre', 'Rating', 'No. of People rated']].iloc[book_indices]
        recommendations.append(recommended_books)

    # Count the frequency of recommended books across all lists
    all_recommended_books = [book for sublist in recommendations for book in sublist['Title']]
    book_counts = Counter(all_recommended_books)

    # Append the count of each recommended book to the books list
    for book in recommendations:
        book['Count'] = book['Title'].apply(lambda x: book_counts[x])

    # Convert the list of recommended books into a 2D array
    recommended_books_array = pd.concat(recommendations, axis=0).reset_index(drop=True)

    # Remove duplicates based on 'Title'
    recommended_books_array = recommended_books_array.drop_duplicates(subset='Title')

    # Reorder the table based on the count from higher to lower
    recommended_books_array = recommended_books_array.sort_values(by='Count', ascending=False)

    return recommended_books_array

# Create the Streamlit web app
st.title('Book Recommendation System')

input_books = st.text_input('Enter book titles separated by commas (,):')
if input_books:
    input_books_list = [book.strip() for book in input_books.split(',')]
    recommended_books = get_recommendations(input_books_list)

    if recommended_books is not None:
	    st.subheader('Recommended Books:')
	    st.write(recommended_books.head())
