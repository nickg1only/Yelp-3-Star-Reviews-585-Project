import json
from tqdm import tqdm

def stream_reviews(review_json, business_json):
    # Streams through reviews.
    # Does NOT load everything at once.
    # This is a Python generator, not a normal function.
    businesses = {json.loads(line)["business_id"]:json.loads(line) for line in open(business_json)}
    for line in open(review_json):
        review = json.loads(line)
        business_id = review["business_id"]
        if "Food" in businesses[business_id]["categories"]:
            yield review
        
        
def stream_pos_reviews(review_json, business_json):
    for review in stream_reviews(review_json, business_json):
        if review["stars"] > 3:
            yield review["text"]

def stream_neg_reviews(review_json, business_json):
    for review in stream_reviews(review_json, business_json):
        if review["stars"] < 3:
            yield review["text"]
            
def stream_neut_reviews(review_json, business_json):
    for review in stream_reviews(review_json, business_json):
        if review["stars"] == 3:
            yield review["text"]
            
            
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
            

            
#############################################################################

            
# Given business ID, return stream of reviews for that business
def stream_business_ids(business_json):
    for line in open(business_json):
        business = json.loads(line)
        if "Food" in business["categories"]:
            yield business["business_id"]
        
def stream_reviews_by_business_id(review_json, business_id):
    for line in open(review_json):
        review = json.loads(line)
        if business_id == review["business_id"]:
            yield review

            
# Write n reviews for a business
def n_reviews_for_a_business(review_json, business_id, n):
    reviews = []
    count = 0
    for review in stream_reviews_by_business_id(review_json, business_id):
        if count >= n:
            break
        reviews.append(review)
        count += 1
    return reviews
            
def all_reviews_for_a_business(review_json, business_id):
    reviews = []
    for review in stream_reviews_by_business_id(review_json, business_id):
        reviews.append(review)
    return reviews


# Returns all reviews for each of n businesses (from start to end)
def all_reviews_for_n_businesses(review_json, business_json, start, end):
    count = 0
    reviews = dict()
    for business_id in stream_business_ids(business_json):
        if count >= end:
            break
        if count >= start and count < end:
            if business_id not in reviews.keys():
                reviews[business_id] = []
            reviews[business_id].extend(all_reviews_for_a_business(review_json, business_id))
        count += 1
    return reviews


# Returns a tuple of positive, neutral, and negative reviews for n businesses
def polarized_reviews_for_n_businesses(review_json, business_json, start, end):
    total_reviews = all_reviews_for_n_businesses(review_json, business_json, start, end)
    polarized_reviews = dict()
    for business_id in total_reviews.keys():
        pos_reviews = [review["text"] for review in total_reviews[business_id] if review["stars"] > 3]
        neut_reviews = [review["text"] for review in total_reviews[business_id] if review["stars"] == 3]
        neg_reviews = [review["text"] for review in total_reviews[business_id] if review["stars"] < 3]
        polarized_reviews[business_id] = (pos_reviews, neut_reviews, neg_reviews)
    return polarized_reviews