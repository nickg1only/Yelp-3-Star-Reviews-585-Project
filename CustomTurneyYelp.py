import math
from ParseYelpData import stream_pos_reviews, stream_neg_reviews
from tqdm import tqdm
import re

import nltk

pos_lex_file = "positive-words.txt"
neg_lex_file = "negative-words.txt"

pos_lexicon = []
neg_lexicon = []

def stream_pos_lexicon(pos_lex_file):
    for i, line in enumerate(open(pos_lex_file)):
        if i >= 35 and line != "" and line != "\n":
            pos_lexicon += line
    pos_lex_file.close()

def stream_neg_lexicon(neg_lex_file):
    for i, line in enumerate(open(neg_lex_file)):
        if i >= 35 and line != "" and line != "\n":
            neg_lexicon += line
    neg_lex_file.close()

def custom_turney():    
    review_json = "../Yelp_dataset/review.json"
    business_json = "../Yelp_dataset/business.json"
    
    # The training set (same as test set) for positive reviews
    pos_training_set = []
    # The training set (same as test set) for negative reviews
    neg_training_set = []

    # Initialize positive training set (which is same as test set)
    pbar = tqdm(total = 100)
    count = 0
    for pos_review in stream_pos_reviews(review_json, business_json):
        if count < 100:
            pos_training_set.append(pos_review)
            pbar.update()
            count += 1
        else:
            pbar.close()
            break

    # Initialize negative training set
    pbar = tqdm(total = 100)
    count = 0
    for neg_review in stream_neg_reviews(review_json, business_json):
        if count < 100:
            neg_training_set.append(neg_review)
            pbar.update()
            count += 1
        else:
            pbar.close()
            break

    # Pseudocount:
    alpha = 0.0001
    
    # Counts of positive and negative seeds, respectively
    num_pos_seeds = 0
    num_neg_seeds = 0
    
    # Set of all words
    nouns = set()
    
    # Lists of all actual positive nouns and actual negative nouns
    actual_pos_nouns = set()
    actual_neg_nouns = set()
    
    # Counts of times a word appears in same sentence as positive or negative reviews, respectively
    num_word_near_pos_dict = dict()
    num_word_near_neg_dict = dict()
    
    # Dict of noun polarity scores
    noun_polarity_scores = dict()
    
    # Noun counts
    noun_counts = dict()

    # List of positive and negative seeds
    stream_pos_lexicon(pos_lex_file)
    stream_neg_lexicon(neg_lex_file)
    
    pos_seeds = pos_lexicon
    neg_seeds = neg_lexicon

    # Parse all positive reviews, one positive review at a time
    pbar2 = tqdm(total = len(pos_training_set))
    for pos_review in pos_training_set:
        
        # Split positive review into sentences separated by spaces
        pos_review_sentences = nltk.sent_tokenize(pos_review)
        
        
        for sentence in pos_review_sentences:
            sentence_without_punctuation = sentence.translate(None, string.punctuation)
            
            # Used for determining if review is positive
            num_pos_seeds_in_sentence = 0
            num_neg_seeds in sentence = 0
            
            # Split sentence into words and parts of speech tags
            words_in_sentence = nltk.pos_tag(sentence_without_punctuation)
            
            # List of nouns in sentence
            nouns_in_sentence = []
            
            # Parse sentence for nouns and positivity measure
            for (word, part_of_speech_tag) in words_in_sentence:
                # Update noun count (both in sentence and in total)
                if part_of_speech_tag in ['NN', 'NNS']:
                    if word not in nouns:
                        noun_counts[word] = 1
                    else:
                        noun_counts[word] += 1
                    nouns_in_sentence += word
                # If positive seed is found, mark sentence as positive and update count of positive sentences
                if word in pos_seeds:
                    num_pos_seeds += 1
                    num_pos_seeds_in_sentence += 1

            # Parse nouns in sentence
            for noun in nouns_in_sentence:
                # Check to make sure word isn't a seed
                if noun not in pos_seeds and word not in neg_seeds:
                    # Update count of times word appears near positive seed
                    if noun not in nouns:
                        nouns.add(noun)
                        actual_pos_nouns.add(noun)
                    num_noun_near_pos_dict[word] = 0
                    num_noun_near_pos_dict[word] += num_pos_seeds_in_sentence
        
        pbar2.update()
    pbar2.close()
    
    # Parse all negative reviews, one negative review at a time
    pbar3 = tqdm(total = len(neg_training_set))
    for neg_review in neg_training_set:
        
        # Split negative review into sentences separated by spaces
        neg_review_sentences = nltk.sent_tokenize(neg_review)
        
        for sentence in neg_review_sentences:
            sentence_without_punctuation = sentence.translate(None, string.punctuation)
            
            # Used for determining if review is negative
            num_neg_seeds_in_sentence = 0
            
            # Split sentence into words and parts of speech tags
            words_in_sentence = nltk.neg_tag(sentence_without_punctuation)
            
            # List of nouns in sentence
            nouns_in_sentence = []
            
            # Parse sentence for nouns and negativity measure
            for (word, part_of_speech_tag) in words_in_sentence:
                # Update noun count (both in sentence and in total)
                if part_of_speech_tag in ['NN', 'NNS']:
                    if word not in nouns:
                        noun_counts[word] = 1
                    else:
                        noun_counts[word] += 1
                    nouns_in_sentence += word
                # If negative seed is found, mark sentence as negative and update count of negative sentences
                if word in neg_seeds:
                    num_neg_seeds += 1
                    num_neg_seeds_in_sentence += 1

            # Parse nouns in sentence
            for noun in nouns_in_sentence:
                # Check to make sure word isn't a seed
                if noun not in pos_seeds and word not in neg_seeds:
                    # Update count of times word appears near negative seed
                    if noun not in nouns:
                        nouns.add(noun)
                        actual_neg_nouns.add(noun)
                    num_noun_near_neg_dict[word] = 0
                    num_noun_near_neg_dict[word] += num_neg_seeds_in_sentence
        
        pbar3.update()
    pbar3.close()
    
    # Calculate word polarity scores
    for noun in nouns:
        if word_counts[noun] >= 500:
            word_polarity_scores[noun] = math.log((num_noun_near_pos_dict[noun] + alpha)*num_neg_seeds*1.0/((num_noun_near_neg_dict[noun] + alpha)*num_pos_seeds*1.0),2)
    
    return word_polarity_scores


word_polarity_scores = custom_turney()
# Initialize positive test set
pos_test_set = []
pbar4 = tqdm(total = 200)
count = 0
for pos_review in stream_pos_reviews(review_json, business_json):
    if count >= 100 and count < 200:
        pos_test_set.append(pos_review)
        pbar4.update()
        count += 1
    elif count < 100:
        pbar4.update()
        count += 1
    else:
        pbar4.close()
        break

# Initialize negative test set
neg_test_set = []
pbar4 = tqdm(total = 200)
count = 0
for neg_review in stream_neg_reviews(review_json, business_json):
    if count >= 100 and count < 200:
        neg_test_set.append(neg_review)
        pbar4.update()
        count += 1
    elif count < 100:
        pbar4.update()
        count += 1
    else:
        pbar4.close()
        break

# 