from textblob import TextBlob
from ParseYelpData import stream_pos_reviews, stream_neg_reviews
from tqdm import tqdm


review_json = "../Yelp_dataset/review.json"
business_json = "../Yelp_dataset/business.json"

def pattern_analyzer(review_json, business_json):
    correct = 0.0
    total = 0.0

    # Classify 100 positive reviews as test set
    pbar = tqdm(total = 100)
    count = 0
    for pos_review in stream_pos_reviews(review_json, business_json):
        if count >= 100:
            break
        if count < 100:
            testimonial = TextBlob(pos_review)
            if testimonial.sentiment.polarity > 0.0:
                correct += 1.0
            total += 1.0
        count += 1
        pbar.update()
    pbar.close()

    # Classify 100 negative reviews as test set
    pbar = tqdm(total = 100)
    count = 0
    for neg_review in stream_neg_reviews(review_json, business_json):
        if count >= 100:
            break
        if count < 100:
            testimonial = TextBlob(neg_review)
            if testimonial.sentiment.polarity <= 0.0:
                correct += 1.0
            total += 1.0
        count += 1
        pbar.update()
    pbar.close()
    
    return 100.0*correct/total


print "Accuracy of TextBlob's Pattern Analyzer:  ", pattern_analyzer(review_json, business_json)

def naive_bayes_analyzer(review_json, business_json):
    from textblob.sentiments import NaiveBayesAnalyzer
    
    correct = 0.0
    total = 0.0
    
    nba = NaiveBayesAnalyzer()

    # Classify 100 positive reviews as test set
    pbar = tqdm(total = 100)
    count = 0
    for pos_review in stream_pos_reviews(review_json, business_json):
        if count >= 100:
            break
        if count < 100:
            testimonial = TextBlob(pos_review, analyzer=nba)
            if testimonial.sentiment.classification == 'pos':
                correct += 1.0
            total += 1.0
        count += 1
        pbar.update()
    pbar.close()

    # Classify 100 negative reviews as test set
    pbar = tqdm(total = 100)
    count = 0
    for neg_review in stream_neg_reviews(review_json, business_json):
        if count >= 100:
            break
        if count < 100:
            testimonial = TextBlob(neg_review, analyzer=nba)
            if testimonial.sentiment.classification != 'pos':
                correct += 1.0
            total += 1.0
        count += 1
        pbar.update()
    pbar.close()
        
    return 100.0*correct/total


print "Accuracy of TextBlob's Naive Bayes Analyzer:  ", naive_bayes_analyzer(review_json, business_json)
