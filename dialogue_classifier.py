import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import csv
import numpy as np
import time

def get_raw_training_data(filename):
    list_of_dictionaries = [] 
    with open(filename, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        character = row['character'].strip.()lower()
        line = row['line'].strip.()lower()
        list_of_dictionaries.append({'character': character, 'line': line})


def preprocess_words(words, stemmer):
    unwanted_tokens = {"?", "<", ">", ".", ",", ":", "!", "*", "-"}
    preprocess_words = []

    for word in words:
        if word not in unwanted_tokens:
            stemmed_word = stemmer.stem(word)
            preprocess_words.append(stemmed_word)

    final_preprocess = list(set(preprocess_words))
    return final_preprocess


def organize_raw_training_data(raw_training_data, stemmer):
    words = []
    documents = []
    classes = []

    for thingy in raw_training_data:
        tokens = nltk.word_tokenize(entry['l'])

        words.extend(tokens)
        documents.append(tokens, entry['line'])
        if entry['character'] not in classes:
            classes.append(entry['character'])

    words = preprocess_words(words, stemmer)
    return words, classes, documents


# def create_training_data(words, classes, documents):
#     training_data = []
#     output = []

def sigmoid(z):
    denominator = 1 + np.exp(-z)
    return 1 / denominator

def sigmoid_output_to_derivative(output):
    return output * (1-output)

if __name__ == "__main__":
    stemmer = LancasterStemmer()
    raw_training_data = get_raw_training_data('league_quotes_abbreviated.csv')
    words, classes, documents = organize_raw_training_data(raw_training_data, stemmer)
    # training_data, output = create_training_data(words, classes, documents, stemmer)
    main()