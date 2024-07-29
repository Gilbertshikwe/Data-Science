## Natural Language Processing (NLP)

Natural Language Processing (NLP) is an interdisciplinary field at the intersection of computer science, artificial intelligence, and linguistics. It focuses on the interaction between computers and humans through natural language, aiming to enable computers to understand, interpret, and generate human language in a meaningful and useful way.

### Key Components of NLP

1. **Tokenization**
   - **Description**: Breaking down text into smaller units called tokens (words, sentences, etc.).
   - **Example**: "I love NLP" becomes `["I", "love", "NLP"]`.

2. **Stop Words Removal**
   - **Description**: Removing common words that do not carry significant meaning (e.g., "and", "the", "is").
   - **Example**: From `["I", "love", "NLP"]`, remove "I".

3. **Stemming and Lemmatization**
   - **Stemming**: Reducing words to their base or root form.
     - **Example**: "running", "runner" -> "run".
   - **Lemmatization**: Reducing words to their base form considering the context.
     - **Example**: "better" -> "good".

4. **Part-of-Speech Tagging (POS)**
   - **Description**: Assigning parts of speech to each word in a sentence.
   - **Example**: "I love NLP" becomes `[("I", "PRP"), ("love", "VBP"), ("NLP", "NNP")]`.

5. **Named Entity Recognition (NER)**
   - **Description**: Identifying entities in text such as names, locations, dates, etc.
   - **Example**: "Google was founded in 1998" -> `[("Google", "ORG"), ("1998", "DATE")]`.

6. **Chunking**
   - **Description**: Grouping related words into chunks.
   - **Example**: "He is a good boy" -> `[("He", "PRP"), ("is a good boy", "NP")]`.

7. **Dependency Parsing**
   - **Description**: Analyzing the grammatical structure of a sentence and establishing relationships between words.
   - **Example**: "She enjoys playing tennis" shows "playing" depends on "enjoys".

8. **Sentiment Analysis**
   - **Description**: Determining the sentiment expressed in a piece of text (positive, negative, neutral).
   - **Example**: "I love this movie!" -> Positive.

9. **Text Classification**
   - **Description**: Categorizing text into predefined classes.
   - **Example**: Spam detection in emails.

10. **Machine Translation**
    - **Description**: Translating text from one language to another.
    - **Example**: Translating "Hello" to "Hola" (English to Spanish).

### Real-Life Applications of NLP

1. **Search Engines**
   - **Description**: Google uses NLP to understand user queries and provide relevant search results.

2. **Chatbots and Virtual Assistants**
   - **Description**: Siri, Alexa, and Google Assistant use NLP to understand and respond to user queries.

3. **Sentiment Analysis**
   - **Description**: Businesses use sentiment analysis to monitor social media, customer reviews, and feedback.

4. **Machine Translation**
   - **Description**: Services like Google Translate use NLP to translate text between languages.

5. **Spam Detection**
   - **Description**: Email services use NLP to detect and filter out spam messages.


# NLP Pipeline Results

This README provides an interpretation of the results obtained from running the `nlp_pipeline.py` script, which demonstrates various key components of Natural Language Processing (NLP) using Python and popular NLP libraries.

## Setup and Requirements

Before running the script, ensure you have the necessary libraries installed:

```bash
pip install nltk spacy textblob scikit-learn googletrans==4.0.0-rc1
python -m spacy download en_core_web_sm
```

## Script Execution

The script processes the example text: 

```
"I love NLP. It's fascinating! Google was founded in 1998."
```

### Results

1. **Tokenization**
   - **Description**: Breaking down text into smaller units called tokens.
   - **Output**:
     ```
     Tokens: ['I', 'love', 'NLP', '.', 'It', "'s", 'fascinating', '!', 'Google', 'was', 'founded', 'in', '1998', '.']
     ```

2. **Stop Words Removal**
   - **Description**: Removing common words that do not carry significant meaning.
   - **Output**:
     ```
     Filtered Tokens: ['love', 'NLP', '.', "'s", 'fascinating', '!', 'Google', 'founded', '1998', '.']
     ```

3. **Stemming and Lemmatization**
   - **Stemming**: Reducing words to their base or root form.
     - **Output**:
       ```
       Stemmed Tokens: ['love', 'nlp', '.', "'s", 'fascin', '!', 'googl', 'found', '1998', '.']
       ```
   - **Lemmatization**: Reducing words to their base form considering the context.
     - **Output**:
       ```
       Lemmatized Tokens: ['love', 'NLP', '.', "'s", 'fascinating', '!', 'Google', 'founded', '1998', '.']
       ```

4. **Part-of-Speech Tagging (POS)**
   - **Description**: Assigning parts of speech to each word in a sentence.
   - **Output**:
     ```
     POS Tags: [('love', 'NN'), ('NLP', 'NNP'), ('.', '.'), ("'s", 'POS'), ('fascinating', 'NN'), ('!', '.'), ('Google', 'NNP'), ('founded', 'VBD'), ('1998', 'CD'), ('.', '.')]
     ```

5. **Named Entity Recognition (NER)**
   - **Description**: Identifying entities in text such as names, locations, dates, etc.
   - **Output**:
     ```
     Named Entities:
     NLP ORG
     Google ORG
     1998 DATE
     ```

6. **Chunking**
   - **Description**: Grouping related words into chunks.
   - **Output**:
     ```
     Chunking: (S
       love/NN
       (ORGANIZATION NLP/NNP)
       ./.
       's/POS
       fascinating/NN
       !/.
       (PERSON Google/NNP)
       founded/VBD
       1998/CD
       ./.)
     ```

7. **Dependency Parsing**
   - **Description**: Analyzing the grammatical structure of a sentence and establishing relationships between words.
   - **Output**:
     ```
     Dependency Parsing:
     I -> nsubj -> love
     love -> ROOT -> love
     NLP -> dobj -> love
     . -> punct -> love
     It -> nsubj -> 's
     's -> ROOT -> 's
     fascinating -> acomp -> 's
     ! -> punct -> 's
     Google -> nsubjpass -> founded
     was -> auxpass -> founded
     founded -> ROOT -> founded
     in -> prep -> founded
     1998 -> pobj -> in
     . -> punct -> founded
     ```

8. **Sentiment Analysis**
   - **Description**: Determining the sentiment expressed in a piece of text.
   - **Output**:
     ```
     Sentiment Analysis: Sentiment(polarity=0.6875, subjectivity=0.7250000000000001)
     ```
   - **Interpretation**: The text has a positive sentiment with a polarity score of 0.6875 and a subjectivity score of 0.725, indicating that the text is quite subjective.

9. **Text Classification**
   - **Description**: Categorizing text into predefined classes.
   - **Output**:
     ```
     Text Classification Prediction: [0]
     ```
   - **Interpretation**: The prediction result `[0]` indicates that the new text "This movie is great" is classified as negative, based on the provided training data.

10. **Machine Translation**
    - **Description**: Translating text from one language to another.
    - **Output**:
      ```
      Machine Translation: Hola
      ```
    - **Interpretation**: The English word "Hello" is correctly translated to the Spanish word "Hola".

## Conclusion

This script provides a comprehensive demonstration of key NLP components, including tokenization, stop words removal, stemming, lemmatization, POS tagging, NER, chunking, dependency parsing, sentiment analysis, text classification, and machine translation. By examining the outputs, we can understand how each component processes and analyzes the given text.
NLP plays a crucial role in enabling computers to understand and interact with human language. Its applications are vast and continue to grow, impacting various industries and enhancing user experiences across different platforms.