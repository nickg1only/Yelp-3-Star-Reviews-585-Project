import json

review_json = "../Yelp_dataset/review.json"

def download_reviews():
    global all_reviews
    all_reviews = open(review_json).readlines()
    return all_reviews
    

business_json = "../Yelp_dataset/business.json"
    
def download_businesses():
    global all_businesses
    all_businesses = open(business_json).readlines()
    return all_businesses
    