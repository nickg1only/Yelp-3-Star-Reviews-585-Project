import json

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