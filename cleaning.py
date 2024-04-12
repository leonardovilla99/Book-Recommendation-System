import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('Books_df.csv')

# Remove duplicates based on the "Title" column
df_cleaned = df.drop_duplicates(subset='Title', keep='first')

# Save the cleaned DataFrame back to a CSV file
df_cleaned.to_csv('cleaned_books_df.csv', index=False)
