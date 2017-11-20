import math
from ParseYelpData import stream_pos_reviews, stream_neg_reviews
from tqdm import tqdm

def word_polarity_scores(file):    
    review_json = "../Yelp_dataset/review.json"
    business_json = "../Yelp_dataset/business.json"
    
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

    # Pseudocount:
    alpha = 0.0001
    
    # Counts of positive and negative seeds, respectively
    num_pos_seeds = 0
    num_neg_seeds = 0
    
    # Set of all words
    words = set()
    
    # Counts of times a word appears in same tweet as positive or negative tweet, respectively
    num_word_near_pos_dict = dict()
    num_word_near_neg_dict = dict()
    
    # Actual word polarity scores
    word_polarity_scores = dict()
    
    # Word counts
    word_counts = dict()

    # List of positive and negative seeds
    pos_seeds = ["good", "nice", "love", "excellent", "fortunate", "correct", "superior"]
    neg_seeds = ["bad", "nasty", "poor", "hate", "unfortunate", "wrong", "inferior"]

    # Parse all tweets, one tweet at a time
    for tweet in tweets_file:
        
        # Split tweet into words separated by spaces
        tweet_words = tweet.split(" ")
        
        # Used for determining if tweet is positive or negative
        num_pos_seeds_in_tweet = 0
        num_neg_seeds_in_tweet = 0
        
        pos_seeds_in_tweet = []
        neg_seeds_in_tweet = []
        
        # Parse tweet for positivity or negativity
        for word in tweet_words:
            # Update word count
            if word not in words:
                word_counts[word] = 1
            else:
                word_counts[word] += 1
            # If positive seed is found, mark tweet as positive and update count of positive tweets
            if word in pos_seeds:
                positive = True
                num_pos_seeds += 1
                num_pos_seeds_in_tweet += 1
            # If negative seed is found, mark tweet as negative and update count of negative tweets
            if word in neg_seeds:
                negative = True
                num_neg_seeds += 1
                num_neg_seeds_in_tweet += 1
                
        # Parse tweet for each word
        for word in tweet_words:
            # Check to make sure word doesn't have @, #, \, or /, and to make sure word isn't a seed
            if word not in pos_seeds and word not in neg_seeds and '@' not in word and '#' not in word:
                # Update count of times word appears near positive/negative seed
                if word not in words:
                    words.add(word)
                    num_word_near_pos_dict[word] = 0
                    num_word_near_neg_dict[word] = 0
                num_word_near_pos_dict[word] += num_pos_seeds_in_tweet
                num_word_near_neg_dict[word] += num_neg_seeds_in_tweet
                
    # Calculate word polarity scores
    for word in words:
        if word_counts[word] >= 500:
            word_polarity_scores[word] = math.log((num_word_near_pos_dict[word] + alpha)*num_neg_seeds*1.0/((num_word_near_neg_dict[word] + alpha)*num_pos_seeds*1.0),2)
    
    return word_polarity_scores