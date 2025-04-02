#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing Basics
#
# This script covers the fundamentals of NLP including:
# 1. Text Preprocessing
# 2. Tokenization
# 3. Word Embeddings
# 4. Vector Operations
# 5. Text Chunking
# 6. Vector Database Basics
# 7. Advanced Word Embeddings with Transformers

# ## 1. Setup and Imports

import nltk
import re
import string
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# ## 2. Text Preprocessing

# Sample text data
sample_text = """
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction 
between computers and human language. It enables computers to understand, interpret, and generate human 
language in a valuable way. NLP combines computational linguistics, machine learning, and deep learning models.
"""

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Apply preprocessing
processed_text = preprocess_text(sample_text)
print("Original Text:")
print(sample_text)
print("\nProcessed Text:")
print(processed_text)

def remove_stopwords(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(filtered_tokens)

# Apply stopword removal
text_without_stopwords = remove_stopwords(processed_text)
print("Text without stopwords:")
print(text_without_stopwords)

def lemmatize_text(text):
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(lemmatized_tokens)

# Apply lemmatization
lemmatized_text = lemmatize_text(text_without_stopwords)
print("Lemmatized text:")
print(lemmatized_text)

# ## 3. Tokenization

def explore_tokenization():
    # Word tokenization
    tokens = word_tokenize(processed_text)
    print(f"Word tokenization (first 10 tokens): {tokens[:10]}")
    
    # Sentence tokenization
    sentences = sent_tokenize(sample_text)
    print(f"\nSentence tokenization (found {len(sentences)} sentences):")
    for i, sentence in enumerate(sentences):
        print(f"Sentence {i+1}: {sentence}")
    
    # N-gram tokenization
    from nltk.util import ngrams
    
    # Generate bigrams
    bigrams = list(ngrams(tokens, 2))
    print(f"\nBigrams (first 5): {bigrams[:5]}")
    
    # Generate trigrams
    trigrams = list(ngrams(tokens, 3))
    print(f"Trigrams (first 5): {trigrams[:5]}")
    
    return tokens, sentences

# Run tokenization
tokens, sentences = explore_tokenization()

# ## 4. Word Embeddings and Vectors

def create_word_embeddings(tokenized_sentences):
    # Using Gensim's Word2Vec
    from gensim.models import Word2Vec
    
    # Train Word2Vec model
    w2v_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    # Find the vector for a word
    print("Vector for 'language':")
    try:
        print(w2v_model.wv['language'][:10], "... (showing first 10 dimensions)")
        
        # Find similar words
        print("\nWords most similar to 'language':")
        similar_words = w2v_model.wv.most_similar('language', topn=5)
        for word, similarity in similar_words:
            print(f"{word}: {similarity:.4f}")
    except KeyError:
        print("Word not in vocabulary. Try with a larger corpus.")
    
    return w2v_model

def create_pytorch_embeddings(tokens):
    # Using PyTorch for embeddings
    import torch
    import torch.nn as nn
    
    # Create a vocabulary
    vocab = list(set(tokens))
    vocab_size = len(vocab)
    embedding_dim = 50
    
    # Create a word to index mapping
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    
    # Create a simple embedding layer
    embedding = nn.Embedding(vocab_size, embedding_dim)
    print(f"Created an embedding layer with {vocab_size} words and {embedding_dim} dimensions")
    
    # Get embedding for a specific word
    word = 'language' if 'language' in word_to_ix else vocab[0]
    word_idx = torch.tensor([word_to_ix[word]])
    word_embedding = embedding(word_idx)
    
    print(f"\nEmbedding for '{word}':")
    print(word_embedding.data[0][:10], "... (showing first 10 dimensions)")
    
    return embedding, word_to_ix, vocab

# Prepare sentences for Word2Vec (each sentence is a list of tokens)
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Create embeddings
w2v_model = create_word_embeddings(tokenized_sentences)
embedding, word_to_ix, vocab = create_pytorch_embeddings(tokens)

# ## 5. Vector Operations

def vector_operations(w2v_model, tokens):
    # Vector similarity with cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    def get_vector(word):
        try:
            return w2v_model.wv[word]
        except KeyError:
            print(f"Word '{word}' not in vocabulary")
            return None
    
    def cosine_sim(word1, word2):
        vec1 = get_vector(word1)
        vec2 = get_vector(word2)
        if vec1 is not None and vec2 is not None:
            # Reshape vectors for sklearn's cosine_similarity
            vec1 = vec1.reshape(1, -1)
            vec2 = vec2.reshape(1, -1)
            return cosine_similarity(vec1, vec2)[0][0]
        return None
    
    # Try some word pairs
    word_pairs = [
        ('language', 'computer'),
        ('natural', 'artificial'),
        ('computer', 'machine')
    ]
    
    print("Cosine similarities between word pairs:")
    for word1, word2 in word_pairs:
        sim = cosine_sim(word1, word2)
        if sim is not None:
            print(f"{word1} - {word2}: {sim:.4f}")
    
    # Vector arithmetic
    # The classic example: king - man + woman = queen
    def vector_equation(pos_words, neg_words):
        # Initialize the result vector with zeros
        result = np.zeros(w2v_model.vector_size)
        
        # Add positive word vectors
        for word in pos_words:
            vec = get_vector(word)
            if vec is not None:
                result += vec
        
        # Subtract negative word vectors
        for word in neg_words:
            vec = get_vector(word)
            if vec is not None:
                result -= vec
        
        return result
    
    # Try a word analogy
    try:
        # Example: language - human + computer = ?
        result_vector = vector_equation(['language', 'computer'], ['human'])
        most_similar = w2v_model.wv.similar_by_vector(result_vector, topn=5)
        
        print("\nlanguage - human + computer = ?")
        for word, similarity in most_similar:
            print(f"{word}: {similarity:.4f}")
    except:
        print("\nCouldn't perform the analogy. Try with a larger corpus or different words.")
    
    return get_vector

# Run vector operations
get_vector = vector_operations(w2v_model, tokens)

# ## 6. Text Chunking and Named Entity Recognition

def explore_chunking():
    # Make sure we have the necessary NLTK data
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    
    # Part-of-speech tagging
    from nltk import pos_tag
    from nltk import ne_chunk
    
    # Sample sentence
    sample_sentence = "Natural Language Processing is developed by researchers at Stanford University and Google."
    tokens = word_tokenize(sample_sentence)
    
    # Apply POS tagging
    tagged = pos_tag(tokens)
    print("Part-of-speech tagging:")
    print(tagged)
    
    # Define a grammar for chunking
    grammar = r"""
        NP: {<DT|PP\$>?<JJ>*<NN>}  # Noun phrase
        VP: {<VB.*><NP|PP>}       # Verb phrase
        PP: {<IN><NP>}            # Prepositional phrase
    """
    
    # Create a chunk parser
    from nltk.chunk import RegexpParser
    chunk_parser = RegexpParser(grammar)
    
    # Apply chunking
    chunks = chunk_parser.parse(tagged)
    
    # Display the chunks
    print("\nChunked text:")
    print(chunks)
    
    # Draw the chunk tree - This would be displayed in a Jupyter notebook
    # chunks.draw()  # Uncomment for notebook
    
    # Named Entity Recognition
    ner_tree = ne_chunk(tagged)
    print("\nNamed Entity Recognition:")
    print(ner_tree)
    
    return chunks, ner_tree

# Run chunking and NER
chunks, ner_tree = explore_chunking()

# ## 7. Simple Vector Database

class SimpleVectorDB:
    def __init__(self):
        self.vectors = {}  # Dictionary to store vectors
    
    def add_vector(self, key, vector):
        """Add a vector to the database"""
        self.vectors[key] = vector
    
    def get_vector(self, key):
        """Retrieve a vector by key"""
        return self.vectors.get(key, None)
    
    def find_similar(self, query_vector, top_k=5):
        """Find the most similar vectors to the query vector"""
        if len(self.vectors) == 0:
            return []
        
        # Calculate cosine similarity for each vector in the database
        similarities = []
        for key, vector in self.vectors.items():
            # Reshape vectors for sklearn's cosine_similarity
            from sklearn.metrics.pairwise import cosine_similarity
            vec1 = query_vector.reshape(1, -1)
            vec2 = vector.reshape(1, -1)
            similarity = cosine_similarity(vec1, vec2)[0][0]
            similarities.append((key, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return similarities[:top_k]

def create_vector_db(w2v_model, tokens):
    # Create and populate our vector database
    vector_db = SimpleVectorDB()
    
    # Add some word vectors from our Word2Vec model
    words_to_add = [word for word in tokens if word in w2v_model.wv]
    for word in words_to_add:
        vector_db.add_vector(word, w2v_model.wv[word])
    
    print(f"Added {len(words_to_add)} vectors to the database")
    
    # Query the database
    query_word = 'language' if 'language' in w2v_model.wv else words_to_add[0]
    query_vector = w2v_model.wv[query_word]
    
    print(f"\nFinding words similar to '{query_word}'")
    similar_words = vector_db.find_similar(query_vector)
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.4f}")
    
    return vector_db

# Create vector database
vector_db = create_vector_db(w2v_model, tokens)

# ## 8. Advanced Word Embeddings with Transformers

def explore_transformer_embeddings():
    # Using HuggingFace transformers library
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    # Load a pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    
    # Sample sentences
    sentences = [
        "The bank is on the river.",  # 'bank' referring to riverbank
        "I went to the bank to deposit money."  # 'bank' referring to financial institution
    ]
    
    # Process the sentences and get embeddings
    def get_sentence_embedding(sentence):
        # Tokenize the sentence
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        
        # Get model output
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the hidden states
        hidden_states = outputs.last_hidden_state
        
        # Use mean pooling to get sentence embedding
        # Mask out padding tokens
        attention_mask = inputs['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        masked_embeddings = hidden_states * mask
        summed = torch.sum(masked_embeddings, dim=1)
        count = torch.sum(attention_mask, dim=1, keepdim=True)
        sentence_embedding = summed / count
        
        return sentence_embedding
    
    # Get embeddings for each sentence
    sentence_embeddings = [get_sentence_embedding(sentence) for sentence in sentences]
    
    # Calculate cosine similarity between the two sentence embeddings
    from torch.nn.functional import cosine_similarity
    similarity = cosine_similarity(sentence_embeddings[0], sentence_embeddings[1]).item()
    
    print(f"Sentences:\n1. {sentences[0]}\n2. {sentences[1]}")
    print(f"Cosine similarity: {similarity:.4f}")
    
    # Extract embeddings for specific words
    def get_word_embedding(sentence, word):
        # Tokenize the sentence
        tokenized = tokenizer(sentence, return_tensors="pt")
        word_tokens = tokenizer.tokenize(word)
        
        # Find the position of the word tokens in the sentence
        input_ids = tokenized.input_ids[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        # Find first occurrence of the word
        word_positions = []
        for i, token in enumerate(tokens):
            if token in word_tokens or token == word:
                word_positions.append(i)
        
        if not word_positions:
            return None
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**tokenized)
        hidden_states = outputs.last_hidden_state[0]
        
        # Get the embedding for the word positions
        word_embedding = hidden_states[word_positions].mean(dim=0)
        return word_embedding
    
    # Get embeddings for 'bank' in both contexts
    bank1 = get_word_embedding(sentences[0], "bank")
    bank2 = get_word_embedding(sentences[1], "bank")
    
    if bank1 is not None and bank2 is not None:
        # Calculate similarity
        bank_similarity = cosine_similarity(bank1.unsqueeze(0), bank2.unsqueeze(0)).item()
        print(f"\nSimilarity between 'bank' in both sentences: {bank_similarity:.4f}")
        print("Note: With contextual embeddings, the meaning of 'bank' differs based on context")
    else:
        print("Could not extract 'bank' embeddings from one or both sentences")

# Uncomment to run transformer embeddings (this may take some time to download and run)
# explore_transformer_embeddings()

# ## 9. Main function to run all sections

def main():
    print("\n" + "="*50)
    print("NLP BASICS DEMONSTRATION")
    print("="*50)
    
    # Text Preprocessing
    print("\n1. TEXT PREPROCESSING")
    processed_text = preprocess_text(sample_text)
    text_without_stopwords = remove_stopwords(processed_text)
    lemmatized_text = lemmatize_text(text_without_stopwords)
    
    # Tokenization
    print("\n2. TOKENIZATION")
    tokens, sentences = explore_tokenization()
    
    # Word Embeddings
    print("\n3. WORD EMBEDDINGS")
    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
    w2v_model = create_word_embeddings(tokenized_sentences)
    embedding, word_to_ix, vocab = create_pytorch_embeddings(tokens)
    
    # Vector Operations
    print("\n4. VECTOR OPERATIONS")
    get_vector = vector_operations(w2v_model, tokens)
    
    # Text Chunking
    print("\n5. TEXT CHUNKING AND NER")
    chunks, ner_tree = explore_chunking()
    
    # Vector Database
    print("\n6. VECTOR DATABASE")
    vector_db = create_vector_db(w2v_model, tokens)
    
    # Transformer Embeddings
    print("\n7. TRANSFORMER EMBEDDINGS")
    print("(Skipped - uncomment this section in the code to run transformer examples)")
    # explore_transformer_embeddings()  # Uncomment to run

if __name__ == "__main__":
    main() 