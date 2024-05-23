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
