import json

def stream_reviews(filename):
    # Streams through reviews.
    # Does NOT load everything at once.
    # This is a Python generator, not a normal function.
    for line in open(filename):
        review = json.loads(line)
        yield review
        
def stream_pos_reviews(filename):
    for review in stream_reviews(filename):
        if review["stars"] > 3:
            yield review["text"]

def stream_neg_reviews(filename):
    for review in stream_reviews(filename):
        if review["stars"] < 3:
            yield review["text"]