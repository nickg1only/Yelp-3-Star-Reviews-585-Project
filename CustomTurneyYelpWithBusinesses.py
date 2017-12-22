import math
from ParseYelpData import all_reviews_for_n_businesses
from tqdm import tqdm
import re
import string
import nltk
                      

            
            
def stream_lexicon(lex_file, lexicon):
    file = open(lex_file)
    for i, line in enumerate(file):
        if i >= 35 and line != "" and line != "\n":
            lexicon.append(line.strip())
    file.close()
    
    

class Turney:
    """ A model implementation of Turney's method for text classification """
    
    
    def __init__(self, jsons, lex_files, alpha):
        # Pseudocount:
        self.alpha = alpha

        # Counts of positive and negative seeds, respectively
        self.num_polarized_seeds = {'pos':0.0, 'neg':0.0}

        # Set of all words
        self.nouns = set()

        # Dict of reviews and the nouns in each of them
        self.review_noun_dict = dict()

        # Lists of all actual positive nouns and actual negative nouns
        self.polarized_nouns = {'pos':set(), 'neg':set()}

        # Counts of times a word appears in same sentence as positive or negative reviews, respectively
        self.num_noun_near_seed_dict = {'pos':dict(), 'neg':dict()}

        # Dict of noun polarity scores
        self.noun_polarity_scores = dict()

        # Noun counts
        self.noun_counts = dict()
        
        self.pos_tags = set()

        # Initialize training sets
        print "Initializing training sets"
        self.pos_training_set, _,self.neg_training_set = polarized_reviews_for_n_businesses(jsons[0], jsons[1], 0, 2) 
        
        # Lists of positive and negative seeds
        self.polarized_seeds = {'pos':[], 'neg':[]}
        print "Streaming lexicons"
        stream_lexicon(lex_files[0], self.polarized_seeds['pos'])
        stream_lexicon(lex_files[1], self.polarized_seeds['neg'])

        # List of all seeds
        self.all_seeds = self.polarized_seeds['pos'] + self.polarized_seeds['neg']

        
    def parse_sentence(self, review, sentence, label):
        # Removes punctuation from sentence
        sentence_without_punctuation = "".join(l for l in sentence if l not in string.punctuation)
        
        # Used for determining if review is positive or negative
        num_polarized_seeds_in_sentence = 0.0
        
        modified_sentence = sentence_without_punctuation.encode('latin1')

        # Split sentence into words and parts of speech tags
        words_in_sentence = modified_sentence.split()
        
        words_and_pos_tags = nltk.pos_tag(words_in_sentence)
        
        
        # List of nouns in sentence
        nouns_in_sentence = []
        
        # Parse sentence for nouns and positivity measure
        for (word, part_of_speech_tag) in words_and_pos_tags:
            
            self.pos_tags.add(part_of_speech_tag)
            
            # Update noun count (both in sentence and in total)
            if part_of_speech_tag in ['NN', 'NNS']:
                if word not in self.nouns:
                    self.noun_counts[word] = 1.0
                else:
                    self.noun_counts[word] += 1.0
                nouns_in_sentence.append(word)
            # If seed is found, update number of (either positive or negative) seeds, both in sentence and in total
            if word in self.polarized_seeds[label]:
                self.num_polarized_seeds[label] += 1.0
                num_polarized_seeds_in_sentence += 1.0

        # Parse nouns in sentence
        for noun in nouns_in_sentence:
            # Check to make sure noun isn't a seed
            if noun not in self.all_seeds:
                # Append noun to review-noun dictionary
                self.review_noun_dict[review].append(noun)
            # Update count of times noun appears near positive seed
            if noun not in self.nouns:
                self.nouns.add(noun)
                self.polarized_nouns[label].add(noun)
            if noun not in self.num_noun_near_seed_dict[label].keys():
                self.num_noun_near_seed_dict[label][noun] = 0.0
            self.num_noun_near_seed_dict[label][noun] += num_polarized_seeds_in_sentence
        
        
    def parse_training_set(self, training_set, label):
        pbar = tqdm(total = len(training_set))
        pbar.set_description("Parsing reviews in %s training set" % label)
        for review in training_set:
            self.review_noun_dict[review] = []
            
            # Split review into sentences separated by spaces
            review_sentences = nltk.sent_tokenize(review)
            for sentence in review_sentences:
                self.parse_sentence(review, sentence, label)
                
            pbar.update()
        pbar.close()
        
    
    def train_model(self):
        print "Parsing positive training set"
        self.parse_training_set(self.pos_training_set, 'pos')
        print "Parsing negative training set"
        self.parse_training_set(self.neg_training_set, 'neg')
    
    
    def calc_polarity_scores(self):
        pbar = tqdm(total = len(self.nouns))
        for noun in self.nouns:
            if noun not in self.num_noun_near_seed_dict['pos'].keys():
                self.num_noun_near_seed_dict['pos'][noun] = 0.0
            if noun not in self.num_noun_near_seed_dict['neg'].keys():
                self.num_noun_near_seed_dict['neg'][noun] = 0.0
            numerator = (self.num_noun_near_seed_dict['pos'][noun]*1.0 + self.alpha)*self.num_polarized_seeds['neg']*1.0
            denominator = (self.num_noun_near_seed_dict['neg'][noun]*1.0 + self.alpha)*self.num_polarized_seeds['pos']*1.0
            ratio = numerator/denominator
            self.noun_polarity_scores[noun] = math.log(ratio,2)
            pbar.update()
        pbar.close()


        
        
# Returns number of reviews turney categorized correctly in the test set
def eval_turney(test_set, polarity, turney_model):
    correct = 0
    pbar3 = tqdm(total = len(test_set))
    for review in test_set:
        review_polarity_score = 0.0
        review_sentences = nltk.sent_tokenize(review)
        turney_model.review_noun_dict[review] = []
        for sentence in review_sentences:
            sentence_without_punctuation = "".join(l for l in sentence if l not in string.punctuation)
            modified_sentence = sentence_without_punctuation.encode('latin1')
            words_in_sentence = modified_sentence.split()
            for word in words_in_sentence:
                if word in turney_model.nouns:
                    turney_model.review_noun_dict[review].append(word)
        for noun in turney_model.review_noun_dict[review]:
            review_polarity_score += turney_model.noun_polarity_scores[noun]
        baseline_score = turney_model.num_polarized_seeds['neg']/turney_model.num_polarized_seeds['pos']
        if polarity == 'pos' and review_polarity_score > baseline_score:
            correct += 1
        if polarity == 'neg' and review_polarity_score < baseline_score:
            correct += 1
        pbar3.update()
    pbar3.close()
    return correct



def top_n_polarized_nouns(turney_model, n):
    sorted_noun_polarity_scores = sorted(turney_model.noun_polarity_scores.items(), key = lambda (k,v): (v,k), reverse = True)
    return (sorted_noun_polarity_scores[:n], sorted_noun_polarity_scores[-n:])

##############################################




# Run custom turney

review_json = "../Yelp_dataset/review.json"
business_json = "../Yelp_dataset/business.json"
jsons = (review_json, business_json)

pos_lex_file = "positive-words.txt"
neg_lex_file = "negative-words.txt"
lex_files = (pos_lex_file, neg_lex_file)

print "Initializing Custom Turney"
turney_model = Turney(jsons, lex_files, 0.0001)


def train_turney():
    turney_model.train_model()
    
    # Calculate polarity scores using Turney's model
    print "Calculating polarity scores"
    turney_model.calc_polarity_scores()

    
def test_turney():
    
    print "Initializing test sets"
    # Initialize positive test set
    pos_test_set = []
    initialize_sample_set(pos_test_set, stream_pos_reviews, jsons, 100, 150)

    # Initialize negative test set
    neg_test_set = []
    initialize_sample_set(neg_test_set, stream_neg_reviews, jsons, 100, 150)

    # Turney's method classifies reviews as positive or negative by adding the word polarity scores in the review
    
    # Evaluate custom turney model
    print "Evaluating turney model"
    # Evaluate on positive test set
    pos_correct = eval_turney(pos_test_set, 'pos', turney_model)
    # Evaluate on negative test set
    neg_correct = eval_turney(neg_test_set, 'neg', turney_model)
    print "Accuracy of Custom Turney: ", 100.0*(pos_correct + neg_correct)/((len(pos_test_set) + len(neg_test_set))*1.0)
    
def print_top_five_polarized_nouns():
    top_five_pos_nouns, top_five_neg_nouns = top_n_polarized_nouns(turney_model, 5)
    print top_five_pos_nouns
    print ""
    print top_five_neg_nouns

def print_noun_polarity_scores():
    print turney_model.noun_polarity_scores
    
def print_part_of_speech_tags():
    print turney_model.pos_tags
    print "NN in pos_tags: %s" % ('NN' in turney_model.pos_tags)
    print "NNS in pos_tags: %s" % ('NNS' in turney_model.pos_tags)
    
def print_nouns():
    print turney_model.nouns