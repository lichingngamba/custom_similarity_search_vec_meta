import logging
from embedding_2 import get_embed

# set logging level
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    corpus = ["The quick brown fox jumps over the lazy dog",
                        "The fox arms are longer than the dog arms", 
                        "Fox is the best animal",
                        "Fox is also the best dog",
                        "The fox and the dog are friends",
                        "The company is named 'Fox and the Dog'",
                        "The lion is the king of the jungle",
                        "Lions are lazy",
                        "Lions are dangerous",
                        "while a lion perfroms in circus, fox dont perform in circus",
                        "Humans are the most intelligent animals"]
    obj = get_embed(corpus= corpus)
    obj.save()

    # check similarity
    relevant_sentence = obj.get_relevant_sentence(sentences= ["Hello, I am the States"])
    for r in relevant_sentence:
        print(f"relevant sentences are: {r}")