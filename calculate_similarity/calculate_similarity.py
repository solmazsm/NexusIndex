#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def calculate_similarity(text1, text2, model_type):
    if model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", padding=True)
        model = BertModel.from_pretrained("bert-base-uncased")
    elif model_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base", padding=True)
        model = RobertaModel.from_pretrained("roberta-base")
    elif model_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding=True)
        model = GPT2Model.from_pretrained("gpt2")
    elif model_type == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", padding=True)
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    
    tokens1 = tokenizer(text1, return_tensors='pt', truncation=True)
    tokens2 = tokenizer(text2, return_tensors='pt', truncation=True)
    
    with torch.no_grad():
        embeddings1 = model(**tokens1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**tokens2).last_hidden_state.mean(dim=1)
    
    similarity = cosine_similarity(embeddings1, embeddings2).item()
    return similarity

