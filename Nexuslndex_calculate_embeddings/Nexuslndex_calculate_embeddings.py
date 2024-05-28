def calculate_embeddings(text, model_type):
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
    
    tokens = tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings.reshape(1, -1)  # Reshape to match the expected shape
