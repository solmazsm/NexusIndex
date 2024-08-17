def index_and_retrieve_embeddings(fake_embeddings, real_embeddings):
   
    index = faiss.IndexFlatL2(fake_embeddings.shape[1])
    index.add(real_embeddings)
    
    # Retrieve similar articles
    def retrieve_similar_articles(query_embedding, top_n=5):
        distances, indices = index.search(query_embedding, top_n)
        return indices[0], distances[0]
    
    return retrieve_similar_articles
