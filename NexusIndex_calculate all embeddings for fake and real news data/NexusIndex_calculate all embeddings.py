def calculate_all_embeddings(fake_data, real_data, model_type):
    fake_embeddings = []
    real_embeddings = []
    for _, row in fake_data.iterrows():
        embedding = calculate_embeddings(row['title'], model_type)
        fake_embeddings.append(embedding)
    for _, row in real_data.iterrows():
        embedding = calculate_embeddings(row['headline_text'], model_type)
        real_embeddings.append(embedding)
    
    fake_embeddings = np.concatenate(fake_embeddings, axis=0)
    real_embeddings = np.concatenate(real_embeddings, axis=0)
    
    return fake_embeddings, real_embeddings
