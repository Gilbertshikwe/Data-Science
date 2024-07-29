import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from googletrans import Translator

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Example text
text = "I love NLP. It's fascinating! Google was founded in 1998."

# 1. Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)

# 2. Stop Words Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("Filtered Tokens:", filtered_tokens)

# 3. Stemming and Lemmatization
# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print("Stemmed Tokens:", stemmed_tokens)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("Lemmatized Tokens:", lemmatized_tokens)

# 4. Part-of-Speech Tagging (POS)
pos_tags = nltk.pos_tag(lemmatized_tokens)
print("POS Tags:", pos_tags)

# 5. Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
print("Named Entities:")
for ent in doc.ents:
    print(ent.text, ent.label_)

# 6. Chunking
ne_tree = ne_chunk(pos_tags)
print("Chunking:", ne_tree)

# 7. Dependency Parsing
print("Dependency Parsing:")
for token in doc:
    print(f'{token.text} -> {token.dep_} -> {token.head.text}')

# 8. Sentiment Analysis
blob = TextBlob(text)
print("Sentiment Analysis:", blob.sentiment)

# 9. Text Classification
# Sample data
texts = ["I love this movie", "I hate this movie"]
labels = [1, 0]  # 1 for positive, 0 for negative

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# Predicting
new_texts = ["This movie is great"]
new_X = vectorizer.transform(new_texts)
predictions = model.predict(new_X)
print("Text Classification Prediction:", predictions)

# 10. Machine Translation
translator = Translator()
translated_text = translator.translate("Hello", dest='es')
print("Machine Translation:", translated_text.text)

