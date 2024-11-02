# importing libraries
import random
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import logging
import numpy as np
import io
import os
import gc

# set logging level
logger = logging.getLogger("embedding")
logger.setLevel(logging.INFO)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set a random seed
random_seed = 42
random.seed(random_seed)

# Set a random seed for PyTorch (for GPU as well)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(random_seed)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

class get_embed():
    def __init__(self, corpus):
        self.example_sentence = corpus
        self.meta_path = r"C:\liching_code\NLP_NBD\vecs_meta\meta_1.tsv"
        self.vec_path = r"C:\liching_code\NLP_NBD\vecs_meta\vecs_1.tsv"
    
    def get_embed(self, sentence): # Get the embedding of the corpase
        encoded =  tokenizer.batch_encode_plus(
                [sentence],
                padding=True,
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            )
        # print("Encoded")
        encoded_token = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        with torch.no_grad():
            output = model(encoded_token, attention_mask = attention_mask).last_hidden_state
            output = output.squeeze(0)
            output = torch.mean(output, dim = 0) 

        # decoded_token = tokenizer.convert_ids_to_tokens(encoded_token[0])
        # output_np = output.detach().numpy()
        return output, sentence # return the embedding, and the sentence

    def run(self): # iterate through all the sentences.
        for sentence in self.example_sentence:
            yield self.get_embed(sentence)

    def save(self): # Save the Document & Embedding as Vector DB
        sent_embed = list()
        for output, sentence in self.run():
            sent_embed.append((sentence, output))
  
        ###
        out_m = io.open(self.meta_path, "w", encoding="utf-8")
        out_v = io.open(self.vec_path, "w", encoding="utf-8")        
        ###
        for i, (sentence, vects) in enumerate(sent_embed):
            out_m.write(sentence + "\n")
            out_v.write('\t'.join([str(x) for x in vects.detach().numpy()]) + "\n")
        out_m.close()
        out_v.close()
    
    # Get top Relevant Vector/Sentence from the Vector DB, Defualt is 2
    # Similarity function used is Cosine Similarity.
    def get_relevant_sentence(self, sentences: list, top_k= 2, func = cosine_similarity):
        store, store_similirity = list(), list()

        for text in sentences:
            output, sentence = self.get_embed(text)
            store.append((sentence, output.detach().numpy()))
        
        s_meta = io.open(self.meta_path, "r", encoding="utf-8")
        s_vecs = io.open(self.vec_path, "r", encoding="utf-8")

        s_meta_lines = s_meta.readlines()
        s_vecs_lines = s_vecs.readlines()

        for i, (sentence, vects) in enumerate(store):

            vects = np.array(vects).reshape(1, -1)
            for r_sentence, r_vects in zip(s_meta_lines, s_vecs_lines):
                r_vects = ' '.join(r_vects.split('\t'))

                r_vects = np.array([float(x) for x in r_vects.split(' ')]).reshape(1, -1)
                
                # check similarity
                similarity = func(r_vects, vects)

                # Store the similarity value
                store_similirity.append((similarity, r_sentence, sentence))

        store_similirity.sort(reverse = True)

        store_similirity = store_similirity[:top_k] # Return only top k
        
        return store_similirity
            
            

