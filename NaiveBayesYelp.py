from ParseYelpData import stream_pos_reviews, stream_neg_reviews

import math
from collections import defaultdict

# Global class labels.
POS_LABEL = 'pos'
NEG_LABEL = 'neg'

def tokenize_review(review):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """
    bow = defaultdict(float)
    tokens = review.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
    return dict(bow)

class NaiveBayes:
    """A Naive Bayes model for text classification."""

    def __init__(self, file, tokenizer):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()
        self.file = file
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
        This function processes the entire training set using the global PATH
        variable above.  It makes use of the tokenize_doc and update_model
        functions you will implement.
        """
        # Tokenize and update the model with 100 positive reviews as training set
        count = 0
        for pos_review in stream_pos_reviews(self.file):
            if count < 100:
                self.tokenize_and_update_model(pos_review, POS_LABEL)
            count += 1
            
        # Tokenize and update the model with 100 negative reviews as training set
        count = 0
        for neg_review in stream_neg_reviews(self.file):
            if count < 100:
                self.tokenize_and_update_model(neg_review, NEG_LABEL)
            count += 1
        
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

    def update_model(self, bow, label):
        """
        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each label (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each label (self.class_total_doc_counts)
        """
        
        for word in bow:
            self.class_word_counts[label][word] += bow[word]
            if (word not in self.vocab):
                self.vocab.add(word)
            self.class_total_word_counts[label] += bow[word]
        self.class_total_review_counts[label] += 1
        

    def tokenize_and_update_model(self, review, label):
        """
        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not

        Make sure when tokenizing to lower case all of the tokens!
        """

        bow = self.tokenize_review(review)
        self.update_model(bow, label)

    def top_n(self, label, n):
        """
        Returns the most frequent n tokens for documents with class 'label'.
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
        Returns the log prior of a document having the class 'label'.
        """
        return math.log((float)(self.class_total_review_counts[label])/sum(self.class_total_review_counts.values()))

    def unnormalized_log_posterior(self, bow, label, alpha):
        """
        Computes the unnormalized log posterior (of doc being of class 'label').
        bow - a bag of words (i.e., a tokenized document)
        """
        return self.log_likelihood(bow, label, alpha) + self.log_prior(label)

    def classify(self, bow, alpha):
        """
        Compares the unnormalized log posterior for doc for both the positive
        and negative classes and returns the either POS_LABEL or NEG_LABEL
        (depending on which resulted in the higher unnormalized log posterior)
        bow - a bag of words (i.e., a tokenized document)
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

        # Classify 100 positive reviews and keep track of how many results are correct
        count = 0
        for pos_review in stream_pos_reviews(file):
            if count >= 100 and count < 200:
                bow = self.tokenize_review(pos_review)
                if self.classify(bow, alpha) == POS_LABEL:
                    correct += 1.0
                total += 1.0
            count += 1
        
        # Classify 100 negative reviews and keep track of how many results are correct
        count = 0
        for neg_review in stream_neg_reviews(file):
            if count >= 100 and count < 200:
                bow = self.tokenize_review(neg_review)
                if self.classify(bow, alpha) == NEG_LABEL:
                    correct += 1.0
                total += 1.0
            count += 1
            
        return 100.0 * correct / total



file = "../Yelp_dataset/review.json"
    
nb = NaiveBayes(file, tokenizer=tokenize_review)
nb.train_model()
print "Accuracy of classifier:  ", nb.evaluate_classifier_accuracy(0.0001)