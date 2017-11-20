from ParseYelpData import stream_pos_reviews, stream_neg_reviews

import math
from collections import defaultdict
from tqdm import tqdm

# Global class labels.
POS_LABEL = 'pos'
NEG_LABEL = 'neg'

def tokenize_review(review):
    """
    Tokenize a review and return its bag-of-words representation.
    review - a string representing a review.
    returns a dictionary mapping each word to the number of times it appears in review.
    """
    bow = defaultdict(float)
    tokens = review.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
    return dict(bow)

class NaiveBayes:
    """A Naive Bayes model for text classification."""

    def __init__(self, review_json, business_json, tokenizer):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()
        # The training set for positive reviews
        self.pos_training_set = []
        # The training set for negative reviews
        self.neg_training_set = []
        # The test set for positive reviews
        self.pos_test_set = []
        # The test set for negative reviews
        self.neg_test_set = []
        
        # Initialize pos training and test sets
        pbar = tqdm(total = 1000)
        pbar2 = tqdm(total = 100)
        count = 0
        for pos_review in stream_pos_reviews(review_json, business_json):
            if count < 1000:
                self.pos_training_set.append(pos_review)
                pbar.update()
            elif count >= 1000 and count < 1100:
                self.pos_test_set.append(pos_review)
                pbar2.update()
            else:
                break
            count += 1
        pbar.close()
        pbar2.close()
        
        # Initialize neg training and test sets
        pbar = tqdm(total = 1000)
        pbar2 = tqdm(total = 100)
        count = 0
        for neg_review in stream_neg_reviews(review_json, business_json):
            if count < 1000:
                self.neg_training_set.append(neg_review)
                pbar.update()
            elif count >= 1000 and count < 1100:
                self.neg_test_set.append(neg_review)
                pbar2.update()
            else:
                break
            count += 1
        pbar.close()
        pbar2.close()
        
        # A JSON file containing the representations of every business in the Yelp Dataset
        self.business_json = business_json
        # A function to tokenize the reviews
        self.tokenize_review = tokenizer
        # class_total_review_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the training set of that class
        self.class_total_review_counts = { POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0 }

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = { POS_LABEL: 0.0,
                                         NEG_LABEL: 0.0 }

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = { POS_LABEL: defaultdict(float),
                                   NEG_LABEL: defaultdict(float) }

    def train_model(self):
        """
        This function processes the entire training set using the global review_json and business_json
        variables above.  It makes use of the tokenize_review and update_model
        functions.
        """
        # Tokenize and update the model with 100 positive reviews as training set
        pbar = tqdm(total = 1000)
        for pos_review in self.pos_training_set:
            self.tokenize_and_update_model(pos_review, POS_LABEL)
            pbar.update()
        pbar.close()
            
            
        # Tokenize and update the model with 100 negative reviews as training set
        pbar = tqdm(total = 1000)
        count = 0
        for neg_review in self.neg_training_set:
            self.tokenize_and_update_model(neg_review, NEG_LABEL)
            pbar.update()
        pbar.close()
        
        # Report some statistics after training the model
        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print "REPORTING CORPUS STATISTICS"
        print "NUMBER OF REVIEWS IN POSITIVE CLASS:", self.class_total_review_counts[POS_LABEL]
        print "NUMBER OF REVIEWS IN NEGATIVE CLASS:", self.class_total_review_counts[NEG_LABEL]
        print "NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL]
        print "NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL]
        print "VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab)
        print "50 MOST COMMON WORDS IN POS REVIEWS:", self.top_n(POS_LABEL, 50)
        print "50 MOST COMMON WORDS IN NEG REVIEWS:", self.top_n(NEG_LABEL, 50)

    def update_model(self, bow, label):
        """
        Update internal statistics given a review represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the review whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each label (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of reviews seen of each label (self.class_total_review_counts)
        """
        
        for word in bow:
            self.class_word_counts[label][word] += bow[word]
            if (word not in self.vocab):
                self.vocab.add(word)
            self.class_total_word_counts[label] += bow[word]
        self.class_total_review_counts[label] += 1
        

    def tokenize_and_update_model(self, review, label):
        """
        Tokenizes a review review and updates internal count statistics.
        review - a string representing a review.
        label - the sentiment of the review (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not

        Make sure when tokenizing to lower case all of the tokens!
        """

        bow = self.tokenize_review(review)
        self.update_model(bow, label)

    def top_n(self, label, n):
        """
        Returns the most frequent n tokens for reviews with class 'label'.
        """
        return sorted(self.class_word_counts[label].items(), key=lambda (w,c): -c)[:n]

    def p_word_given_label(self, word, label):
        """
        Returns the probability of word given label
        according to this NB model.
        """
        return (float)(self.class_word_counts[label][word])/(self.class_total_word_counts[label])

    def p_word_given_label_and_pseudocount(self, word, label, alpha):
        """
        Returns the probability of word given label wrt psuedo counts.
        alpha - pseudocount parameter
        """
        return (float)(self.class_word_counts[label][word] + alpha)/(self.class_total_word_counts[label] + alpha*len(self.vocab))

    def log_likelihood(self, bow, label, alpha):
        """
        Computes the log likelihood of a set of words give a label and pseudocount.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; pseudocount parameter
        """
        log_likelihood = 0.0
        for word in bow:
            log_likelihood += math.log(self.p_word_given_label_and_pseudocount(word, label, alpha))
        return log_likelihood

    def log_prior(self, label):
        """
        Returns the log prior of a review having the class 'label'.
        """
        return math.log((float)(self.class_total_review_counts[label])/sum(self.class_total_review_counts.values()))

    def unnormalized_log_posterior(self, bow, label, alpha):
        """
        Computes the unnormalized log posterior (of review being of class 'label').
        bow - a bag of words (i.e., a tokenized review)
        """
        return self.log_likelihood(bow, label, alpha) + self.log_prior(label)

    def classify(self, bow, alpha):
        """
        Compares the unnormalized log posterior for review for both the positive
        and negative classes and returns the either POS_LABEL or NEG_LABEL
        (depending on which resulted in the higher unnormalized log posterior)
        bow - a bag of words (i.e., a tokenized review)
        """
        pos_ulp = self.unnormalized_log_posterior(bow, POS_LABEL, alpha)
        neg_ulp = self.unnormalized_log_posterior(bow, NEG_LABEL, alpha)
        
        if pos_ulp > neg_ulp:
            return POS_LABEL
        else:
            return NEG_LABEL

    def likelihood_ratio(self, word, alpha):
        """
        Returns the ratio of P(word|pos) to P(word|neg).
        """
        return self.p_word_given_label_and_pseudocount(word, POS_LABEL, alpha)/self.p_word_given_label_and_pseudocount(word, NEG_LABEL, alpha)

    def evaluate_classifier_accuracy(self, alpha):
        """
        alpha - pseudocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right.
        """
        correct = 0.0
        total = 0.0

        # Classify 100 positive reviews (test set) and keep track of how many results are correct
        pbar = tqdm(total = 100)
        for pos_review in self.pos_test_set:
            bow = self.tokenize_review(pos_review)
            if self.classify(bow, alpha) == POS_LABEL:
                correct += 1.0
            total += 1.0
            pbar.update()
        pbar.close()
        
        # Classify 100 negative reviews (test set) and keep track of how many results are correct
        pbar = tqdm(total = 100)
        count = 0
        for neg_review in self.neg_test_set:
            bow = self.tokenize_review(neg_review)
            if self.classify(bow, alpha) == NEG_LABEL:
                correct += 1.0
                total += 1.0
            pbar.update()
        pbar.close()
            
        return 100.0 * correct / total


review_json = "../Yelp_dataset/review.json"
business_json = "../Yelp_dataset/business.json"

nb = NaiveBayes(review_json, business_json, tokenizer=tokenize_review)

def train_NB_model():
    nb.train_model()

def test_NB_model():
    # Set pseudocount to 0.0001
    print "Accuracy of classifier:  ", nb.evaluate_classifier_accuracy(0.0001)