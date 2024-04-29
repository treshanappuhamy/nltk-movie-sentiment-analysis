import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

# Download NLTK resources (uncomment the following line if not downloaded)
# nltk.download('movie_reviews')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Load movie reviews dataset
positive_reviews = [(movie_reviews.words(fileids=[f]), 'pos') for f in movie_reviews.fileids('pos')]
negative_reviews = [(movie_reviews.words(fileids=[f]), 'neg') for f in movie_reviews.fileids('neg')]
all_reviews = positive_reviews + negative_reviews

# Define function for text preprocessing
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return dict([(token, True) for token in cleaned_tokens])

# Preprocess and prepare feature sets
feature_sets = [(preprocess_text(words), sentiment) for (words, sentiment) in all_reviews]

# Split feature sets into training and testing sets
train_set, test_set = feature_sets[:1600], feature_sets[1600:]

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate the classifier
print("Accuracy:", accuracy(classifier, test_set))

# Example usage
test_sentence = "This movie is fantastic!"
preprocessed_test_sentence = preprocess_text(test_sentence)
print("Sentiment:", classifier.classify(preprocessed_test_sentence))
