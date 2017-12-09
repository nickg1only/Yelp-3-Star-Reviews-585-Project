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
            
#def reviews_by_business(jsons, business_id):
    