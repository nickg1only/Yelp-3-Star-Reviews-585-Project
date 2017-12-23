import json

##########################################################


def parsed_reviews(start, end, all_reviews):
    return [json.loads(all_reviews[i]) for i in range(start, end)]
    
def print_review_texts(start, end, all_reviews):
    reviews = parsed_reviews(start, end, all_reviews)
    for i in range(len(reviews)):
        print reviews[i]["text"]
        
##########################################################

def parse_businesses(start, end, all_businesses):
    return [json.loads(all_businesses[i]) for i in range(start, end)]

#############################################################
# BAD FUNCTIONS
business_ids = []
def parse_all_business_ids():
    business_ids = [business["business_id"] for business in all_businesses]
    
def get_business_ids(indices):
    return [business_ids[index] for index in indices]

###############################################################
# GOOD FUNCTION
def reviews_by_business_id(start, end, all_reviews, business_id):
    return [review for review in parsed_reviews(start, end, all_reviews) if review["business_id"].encode("latin-1") == business_id]

def neut_reviews_by_business_id(start, end, all_reviews, business_id):
    return [review["text"] for review in reviews_by_business_id(start, end, all_reviews, business_id) if reviews["stars"] == 3]

###########################################################################
# GOOD FUNCTION

def print_business_ids(start, end, all_businesses):
    businesses = parse_businesses(start, end, all_businesses)
    for i in range(len(businesses)):
        print businesses[i]["business_id"]
