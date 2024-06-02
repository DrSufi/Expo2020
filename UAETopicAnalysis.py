import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the data
data = pd.read_csv('C:\\Users\\USER\Documents\\Research\\Development\\uae4.csv')

# Convert the 'Time' column to datetime format
data['Time'] = pd.to_datetime(data['Time'])

# Create a new column for the month
data['Month'] = data['Time'].dt.to_period('M')

# Create a new column 'FinalText' based on the condition you specified
data['FinalText'] = data.apply(lambda row: row['TranslatedText'] if pd.notna(row['TranslatedText']) and row['TranslatedText'] != '' else row['TweetText'], axis=1)

# Tokenization, stop word removal, and lemmatization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

data['ProcessedText'] = data['FinalText'].apply(preprocess_text)

# Split the data into train and test sets (for simplicity, you can adjust the split ratio)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['ProcessedText'])

# Latent Dirichlet Allocation (LDA) for topic modeling
num_topics = 5  # Adjust as needed
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(tfidf_matrix)

# Get the topics and associated words
feature_names = tfidf_vectorizer.get_feature_names_out()
topics = {}
for topic_idx, topic in enumerate(lda_model.components_):
    topics[f'Topic {topic_idx + 1}'] = [feature_names[i] for i in topic.argsort()[:-11:-1]]

# Display the topics and associated words
for topic, words in topics.items():
    print(f'{topic}: {", ".join(words)}')

# Transform the test data
test_tfidf_matrix = tfidf_vectorizer.transform(test_data['ProcessedText'])
test_topics = lda_model.transform(test_tfidf_matrix)

# Add the predicted topic distribution to the test data
for i in range(num_topics):
    test_data[f'Topic {i + 1}'] = test_topics[:, i]

# Display the results for the test data
print("\nTest Data with Predicted Topics:")
print(test_data[['Time', 'ProcessedText'] + [f'Topic {i + 1}' for i in range(num_topics)]])
