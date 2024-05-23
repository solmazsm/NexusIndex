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
