"""
PQ3 -- Dialogue Classifier
James McGowan
Emre Andican
Afamdi Achufusi
"""

import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import csv
import numpy as np
import time
from PQ3 import start_training, classify

nltk.download('punkt_tab')

def get_raw_training_data(filename):
    list_of_dictionaries = [] 
    with open(filename, newline='') as csvfile:
        # Specify the column names manually since the file has no headers
        reader = csv.DictReader(csvfile, fieldnames=['character', 'line'])
        for row in reader:
            # Process each row
            character = row['character'].strip().lower()
            line = row['line'].strip().lower()
            list_of_dictionaries.append({'character': character, 'line': line})
    return list_of_dictionaries



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
        tokens = nltk.word_tokenize(thingy['line'])

        words.extend(tokens)
        documents.append((tokens, thingy['character']))
        if thingy['character'] not in classes:
            classes.append(thingy['character'])

    words = preprocess_words(words, stemmer)
    return words, classes, documents


def create_training_data(words, classes, documents, stemmer):
    training_data = []
    output = []

    output_col = [0] * len(classes)

    for token, character in documents:
        stems = preprocess_words(token, stemmer)
        
        bag = []
        for word in words:
            if word in stems:
                bag.append(1)
            else:
                bag.append(0)
        training_data.append(bag)
    
        # Not sure if this right terminology/process
        output_row = output_col[:]
        output_row[classes.index(character)] = 1
        output.append(output_row)

    return training_data, output


def sigmoid(z):
    denominator = 1 + np.exp(-z)
    return 1 / denominator


def sigmoid_output_to_derivative(output):
    return output * (1-output)


def main():
    stemmer = LancasterStemmer()
    raw_training_data = get_raw_training_data('league_quotes_abbreviated.csv')
    words, classes, documents = organize_raw_training_data(raw_training_data, stemmer)
    training_data, output = create_training_data(words, classes, documents, stemmer)

    start_training(words, classes, training_data, output)
    
    sentences_to_classify = [
        "The power of the wild is before you!",
        "The dark star hungers for your soul.",
        "Ow...my groove...",
        "Go U Bears! We are the storm.",
        "Blood for the blood god!",
        "No soul can escape me.",
        "Party time. Heheheha!",
        "Hi, little guy. This'll be messy.",
        "I'm armed and ready. Nobody escapes. Any last words?",
        "Yes! Yes! Yes! Let the storm follow in my wake. Yes!"
    ]

    for sentence in sentences_to_classify:
        classify(words, classes, sentence)


if __name__ == "__main__":
    main()