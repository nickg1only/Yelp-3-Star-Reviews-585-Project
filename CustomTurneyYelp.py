import math
from ParseYelpData import stream_pos_reviews, stream_neg_reviews
from tqdm import tqdm
import re
import string
import nltk



def stream_lexicon(lex_file, lexicon):
    file = open(lex_file)
    for i, line in enumerate(file):
        if i >= 35 and line != "" and line != "\n":
            lexicon.append(line)
    file.close()
    
    
    
def initialize_sample_set(sample_set, stream_function, jsons, start, finish):
    pbar = tqdm(total = (finish-start))
    count = 0
    for review in stream_function(jsons[0], jsons[1]):
        if count >= start and count < finish:
            sample_set.append(review)
            pbar.update()
            count += 1
        elif count < start:
            count += 1
        else:
            pbar.close()
            break                          

            
            
def parse_reviews(training_set, total_seeds, seeds, num_seeds, nouns, review_noun_dict, actual_polarized_nouns, num_noun_near_seed_dict, noun_counts):
    # Parse all reviews in training set, one review at a time
    pbar2 = tqdm(total = len(training_set))
    for review in training_set:
        
        review_noun_dict[review] = []
        
        # Split review into sentences separated by spaces
        review_sentences = nltk.sent_tokenize(review)
        
        # Parse sentence
        for sentence in review_sentences:
            # Removes punctuation from sentence
            sentence_without_punctuation = "".join(l for l in sentence if l not in string.punctuation)
            
            # Used for determining if review is positive or negative
            num_seeds_in_sentence = 0
            
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
                    nouns_in_sentence.append(word)
                # If seed is found, update number of (either positive or negative) seeds, both in sentence and in total
                if word in seeds:
                    num_seeds += 1
                    num_seeds_in_sentence += 1

            # Parse nouns in sentence
            for noun in nouns_in_sentence:
                # Check to make sure noun isn't a seed
                if noun not in total_seeds:
                    # Append noun to review-noun dictionary
                    review_noun_dict[review].append(noun)
                    # Update count of times noun appears near positive seed
                    if noun not in nouns:
                        nouns.add(noun)
                        actual_polarized_nouns.add(noun)
                    if noun not in num_noun_near_seed_dict.keys():
                        num_noun_near_seed_dict[noun] = 0
                    num_noun_near_seed_dict[noun] += num_seeds_in_sentence
        
        pbar2.update()
    pbar2.close()

    

def custom_turney(jsons, lex_files):    
    
    pos_training_set = [] # The training set for positive reviews
    neg_training_set = [] # The training set for negative reviews

    print "Initializing training sets"
    initialize_sample_set(pos_training_set, stream_pos_reviews, jsons, 0, 100) # Initialize positive training set
    initialize_sample_set(neg_training_set, stream_neg_reviews, jsons, 0, 100) # Initialize negative training set

    # Pseudocount:
    alpha = 0.0001
    
    # Counts of positive and negative seeds, respectively
    num_pos_seeds = 0
    num_neg_seeds = 0
    
    # Set of all words
    nouns = set()
    
    # Dict of reviews and the nouns in each of them
    review_noun_dict = dict()
    
    # Lists of all actual positive nouns and actual negative nouns
    actual_pos_nouns = set()
    actual_neg_nouns = set()
    
    # Counts of times a word appears in same sentence as positive or negative reviews, respectively
    num_noun_near_pos_dict = dict()
    num_noun_near_neg_dict = dict()
    
    # Dict of noun polarity scores
    noun_polarity_scores = dict()
    
    # Noun counts
    noun_counts = dict()

    # List of positive and negative seeds
    pos_seeds = []
    neg_seeds = []
    print "Streaming lexicons"
    stream_lexicon(lex_files[0], pos_seeds)
    stream_lexicon(lex_files[1], neg_seeds)
    
    total_seeds = pos_seeds + neg_seeds

    print "Parsing training sets"
    # Parse positive training set
    parse_reviews(pos_training_set, total_seeds, pos_seeds, num_pos_seeds, nouns, review_noun_dict, actual_pos_nouns, num_noun_near_pos_dict, noun_counts)
    # Parse negative training set
    parse_reviews(neg_training_set, total_seeds, neg_seeds, num_neg_seeds, nouns, review_noun_dict, actual_neg_nouns, num_noun_near_neg_dict, noun_counts)
    
    print "Calculating polarity scores"
    # Calculate noun polarity scores
    for noun in nouns:
        if noun not in num_noun_near_pos_dict.keys():
            num_noun_near_pos_dict[noun] = 0.0
        if noun not in num_noun_near_neg_dict.keys():
            num_noun_near_neg_dict[noun] = 0.0
        numerator = (num_noun_near_pos_dict[noun] + alpha)*num_neg_seeds*1.0
        denominator = (num_noun_near_neg_dict[noun] + alpha)*num_pos_seeds*1.0
        print "Numerator: ", numerator
        print "Denominator: ", denominator
        print "Num pos seeds: ", num_pos_seeds
        print "Num neg seeds: ", num_neg_seeds
        print "Alpha: ", alpha
        print "Num nouns near pos: ", num_noun_near_pos_dict[noun]
        print "Num nouns near neg: ", num_noun_near_neg_dict[noun]
        ratio = numerator/denominator
        noun_polarity_scores[noun] = math.log(ratio,2)
    
    return review_noun_dict, noun_polarity_scores, actual_pos_nouns, actual_neg_nouns



# Returns number of reviews turney categorized correctly in the test set
def eval_turney(test_set, review_noun_dict, noun_polarity_scores):
    correct = 0
    pbar3 = tqdm(total = len(test_set))
    for review in test_set:
        review_polarity_score = 0
        for noun in review_noun_dict[review]:
            review_polarity_score += noun_polarity_scores[noun]
        if review_polarity_score > 0:
            correct += 1
    return correct




##############################################




# Run custom turney

review_json = "../Yelp_dataset/review.json"
business_json = "../Yelp_dataset/business.json"
jsons = (review_json, business_json)

pos_lex_file = "positive-words.txt"
neg_lex_file = "negative-words.txt"
lex_files = (pos_lex_file, neg_lex_file)

print "Running Custom Turney"
review_noun_dict, noun_polarity_scores, actual_pos_nouns, actual_neg_nouns = custom_turney(jsons, lex_files)


print "Initializing test sets"
# Initialize positive test set
pos_test_set = []
initialize_sample_set(pos_test_set, stream_pos_reviews, jsons, 100, 150)

# Initialize negative test set
neg_test_set = []
initialize_sample_set(neg_test_set, stream_neg_reviews, jsons, 100, 150)


# Turney's method classifies reviews as positive or negative by adding the word polarity scores in the review

# Evaluate custom turney

# Evaluate on positive test set
pos_correct = eval_turney(pos_test_set, review_noun_dict, noun_polarity_scores)
neg_correct = eval_turney(neg_test_set, review_noun_dict, noun_polarity_scores)
print "Accuracy of Custom Turney: ", 100.0*(pos_correct + neg_correct)/(len(pos_test_set) + len(neg_test_set))