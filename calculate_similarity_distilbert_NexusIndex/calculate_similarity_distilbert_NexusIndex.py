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
