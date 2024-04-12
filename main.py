import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Load the dataset
data = pd.read_csv('Books_df.csv')

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
        # Get the index of the book that matches the title
        idx = data[data['Title'] == title].index[0]

        # Get the pairwise similarity scores of all books with that book
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the books based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the top 10 most similar books
        sim_scores = sim_scores[1:51]  # Change to 10 recommendations

        # Get the book indices
        book_indices = [i[0] for i in sim_scores]

        # Get the top 10 most similar books and their scores
        recommended_books = data[['Title', 'Author', 'Main Genre', 'Sub Genre', 'Rating', 'No. of People rated']].iloc[book_indices]

        # Remove books with the same title as the input title
        recommended_books = recommended_books[recommended_books['Title'] != title]

        recommendations.append(recommended_books)


    # Count the frequency of recommended books across all lists
    all_recommended_books = [book for sublist in recommendations for book in sublist['Title']]
    book_counts = Counter(all_recommended_books)

    # Append the count of each recommended book to the books list
    for book in recommendations:
        book['Count'] = book['Title'].apply(lambda x: book_counts[x])

    # Convert the list of recommended books into a 2D array
    recommended_books_array = pd.concat(recommendations, axis=0).reset_index(drop=True)

    # Remove duplicates if any
    recommended_books_array = recommended_books_array.drop_duplicates(subset='Title')

    # Reorder the table based on the count from higher to lower
    recommended_books_array = recommended_books_array.sort_values(by='Count', ascending=False)

    return recommended_books_array

# Example usage
input_books = ["Dune","1984"]
recommended_books_array = get_recommendations(input_books)

# Print the reordered and duplicate-free 2D array of recommended books
print("Recommended Books:")
print(recommended_books_array.head())
