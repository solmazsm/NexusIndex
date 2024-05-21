#cosine similarity
#!/usr/bin/env python
# coding: utf-8



def calculate_similarity_gpt2(text1, text2):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")

    tokens1 = tokenizer(text1, return_tensors='pt', truncation=True)
    tokens2 = tokenizer(text2, return_tensors='pt', truncation=True)

    with torch.no_grad():
        embeddings1 = model(**tokens1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**tokens2).last_hidden_state.mean(dim=1)

    # Manually add padding if needed
    if embeddings1.shape[1] < embeddings2.shape[1]:
        padding = torch.zeros_like(embeddings1[:, :1, :])
        embeddings1 = torch.cat([embeddings1, padding], dim=1)
    elif embeddings2.shape[1] < embeddings1.shape[1]:
        padding = torch.zeros_like(embeddings2[:, :1, :])
        embeddings2 = torch.cat([embeddings2, padding], dim=1)

    similarity = cosine_similarity(embeddings1, embeddings2).item()
    return similarity




def calculate_similarity_roberta(text1, text2):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")

    tokens1 = tokenizer(text1, return_tensors='pt', truncation=True)
    tokens2 = tokenizer(text2, return_tensors='pt', truncation=True)

    with torch.no_grad():
        embeddings1 = model(**tokens1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**tokens2).last_hidden_state.mean(dim=1)

    # Manually add padding if needed
    if embeddings1.shape[1] < embeddings2.shape[1]:
        padding = torch.zeros_like(embeddings1[:, :1, :])
        embeddings1 = torch.cat([embeddings1, padding], dim=1)
    elif embeddings2.shape[1] < embeddings1.shape[1]:
        padding = torch.zeros_like(embeddings2[:, :1, :])
        embeddings2 = torch.cat([embeddings2, padding], dim=1)

    similarity = cosine_similarity(embeddings1, embeddings2).item()
    return similarity


def calculate_similarity_bert(text1, text2):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", padding=True)
    model = BertModel.from_pretrained("bert-base-uncased")

    tokens1 = tokenizer(text1, return_tensors='pt', truncation=True)
    tokens2 = tokenizer(text2, return_tensors='pt', truncation=True)

    with torch.no_grad():
        embeddings1 = model(**tokens1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**tokens2).last_hidden_state.mean(dim=1)

    similarity = cosine_similarity(embeddings1, embeddings2).item()
    return similarity


def calculate_similarity_distilbert(text1, text2):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    tokens1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True)
    tokens2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        embeddings1 = model(**tokens1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**tokens2).last_hidden_state.mean(dim=1)

    similarity = cosine_similarity(embeddings1, embeddings2).item()
    return similarity


subset_real_data = real_news_df[['publish_date', 'headline_text']].sample(n=10)
subset_fake_data = fake_news_df[['title', 'text', 'label']].sample(n=10)

subset_real_data['label'] = 1


similarity_list = []

for i in range(len(subset_fake_data)):
    for j in range(len(subset_real_data)):
        headline_fake = subset_fake_data['title'].iloc[i]
        headline_real = subset_real_data['headline_text'].iloc[j]

       
        similarity_bert = calculate_similarity_bert(headline_fake, headline_real)
        similarity_roberta = calculate_similarity_roberta(headline_fake, headline_real)
        similarity_gpt2 = calculate_similarity_gpt2(headline_fake, headline_real)
        similarity_distilbert = calculate_similarity_distilbert(headline_fake, headline_real)

       
        similarity_list.append({
            'Fake News Headline': headline_fake,
            'Real News Headline': headline_real,
            'Similarity (BERT)': similarity_bert,
            'Similarity (RoBERTa)': similarity_roberta,
            'Similarity (GPT-2)': similarity_gpt2,
            'Similarity (DistilBERT)': similarity_distilbert,
            'Fake News Label': subset_fake_data['label'].iloc[i],
            'Real News Label': subset_real_data['label'].iloc[j]
        })

