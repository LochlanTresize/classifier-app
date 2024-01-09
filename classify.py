
from collections import defaultdict as dd
import csv
from math import sqrt
import pickle

# We expect to deal with large csv files when training the model, so adjust the csv field size limit
csv.field_size_limit(100000000)

def count_trigrams(document):
    '''
    count_trigrams takes a string and returns a dictionary of the counts 
    of trigrams within the document (the language vector)
    the input 'document' is a string
    '''

    language_vector = dd(int)

    if len(document) < 3:
        return language_vector
    
    for i in range(len(document) - 2):
        language_vector[document[i] + document[i+1] + document[i+2]] += 1

    return language_vector


# todo in future: adapt this code to calculate n-grams so we can improve the accuracy of the model by optimising n.


def normalise(language_vector):
    '''
    normalise the values in a dictionary of trigram counts, returning them in a dictionary
    '''
    #calculate the magnitude of the dictionaries values (each value is taken to be a scaled basis vector, with each basis vector being a unique trigram)
    # each trigram is taken to be a basis vector in the vector space of a language's trigrams.
    # so to normalise the trigrams we scale all vectors in trigram_counts 
    magnitude = sqrt(sum(x**2 for x in language_vector.values()))

    for key in language_vector:
        language_vector[key] /= magnitude

    return language_vector



# ## Creating language vectors
# 
# We now write a function to read texts from csv files in a training directory. Recall the csv files all have format language, text1, text2. We will return a dictionary of languages with their normalised trigram counts (or normalised language vectors) in the form *{language1: trigram_counts1, Language2: trigram_counts2, ...}*.
# 
# In future we should try and clean data to better train the model on - e.g. remove trigrams found in links, code, etc.



def train_classifier(training_directory):
    '''
    takes list of file names to train the model on and returns a dictionary of normalised language vectors.
    '''
    language_vectors = dd()     # the dictionary with all language vectors


    for filename in training_directory:
        with open(filename, 'r', encoding='utf8') as fp:
            for line in csv.reader(fp):
                line = list(line)

                # use .upper() so we don't get duplicate languages 
                language = line[0].upper()

                # add trigrams counts to this vector
                create_language_vector = dd(int)

                for text in line[1:]:
                    create_language_vector.update(count_trigrams(text))

                # check if we have seen this language before (i.e. whether the language vector exists or not)
                if language in language_vectors:
                    # if the language vector exists add it to its existing language vector
                    for trigram in create_language_vector:
                        language_vectors[language][trigram] += create_language_vector[trigram]
                else:
                    language_vectors[language] = create_language_vector
    

    # don't normalise vectors until now in case there is more texts in an alreay seen language later in the training data
    for vector in language_vectors:
        language_vectors[vector] = normalise(language_vectors[vector])
    

    return language_vectors

# ## Scoring Documents
# We now move on to scoring input documents (performing the above mentioned euclidean inner product).
# 
# In future, the below algorithm should be made more efficient, as it is currently somewhat slow. For example, representing data in matrices rather than dictionaries could help speed up matrix multiplication in the inner product calculation. Something similar should also be looked at to try and avoid looping through trained_trigrams, as what happens at the bottom of the above code block.


def score_document(document_to_classify, language_vectors):
    '''
    takes in a document to classify (string) and the dictionary of language vectors and returns the scores for each language in language vectors
    '''


    document_to_classify_trigrams = count_trigrams(document_to_classify)
    languages_scores = dd(int)
        
    for vector in language_vectors:
        current_language_vector = language_vectors[vector]
            
        for trigram in document_to_classify_trigrams:
            if trigram in current_language_vector:
                new_score = document_to_classify_trigrams[trigram] * current_language_vector[trigram]
                languages_scores[vector] += new_score
    
    
    return languages_scores

# ## Classifying Documents
# We now implement a method to classify the language a document has been written in.
# 
# We do this by finding the language with the highest score in languages_scores and returning it.
# 
# To do this we sort the keys in score_document according to the magnitude of their values. We use a tolerance of 1e-10. Should a tie exist in this range, we return both possible languages and ask the user to enter more text.
# 
# In future we should allow the user to list out the k most likely languages.


def classify_doc(document_to_classify, language_vectors, tolerance, classifying_file=False):
    '''
    The document to classify can be either a string (classifying_file=False by default) or a text file (classifying_file=true)
    language_vectors is the dictionary of language vectors
    
    tolerance the min difference in scores two languages can have and be given different ranks
    returns the language a text is written in, or, if a tie exists, a list of possible languages and a prompt to enter more text
    '''    

    if classifying_file:
        document_to_classify = open(str(document_to_classify)).read()

    # returns dictionary of all languages scores
    document_to_classify_scores = score_document(document_to_classify, language_vectors)
    
    # initialise the most_common_language to be any of the languages
    most_common_language = list(document_to_classify_scores.keys())[0]
    
    tie = [most_common_language]

    
    for language in document_to_classify_scores:
        if (document_to_classify_scores[most_common_language] - document_to_classify_scores[language]) > 1e-10:
            continue
        elif (document_to_classify_scores[most_common_language] - document_to_classify_scores[language]) < -(1e-10):
            most_common_language = language
            tie = [most_common_language]
        else:
            tie.append(language)

    # if no tie
    if len(set(tie)) == 1:
        return most_common_language

    else:
        print('There is a tie. The possible languages are as follows:')
        print(tie)
        print('For a more accurate model, please input more text')


with open('trained_data.pkl', 'rb') as newfpx:
    trained_data = pickle.load(newfpx)


def classify_text(string):
    return classify_doc(string, trained_data, 1e-10, classifying_file=False)

def info():
    return len(trained_data.keys())
# print(f'Languages trained on: {[key for key in trained_data.keys()]}')

