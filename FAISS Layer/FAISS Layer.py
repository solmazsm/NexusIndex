;==========================================
; Title: NexusIndexModel-FAISSLayer
; Author: Solmaz Seyed Monir
;==========================================
class FAISSLayer(nn.Module):
    def __init__(self, d):
        super(FAISSLayer, self).__init__()
        self.index = faiss.IndexFlatL2(d)
    
    def forward(self, x, add_to_index=False):
        embeddings_np = x.detach().cpu().numpy()
        if embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)
        assert embeddings_np.shape[1] == self.index.d, f"Expected embedding dimension {self.index.d}, but got {embeddings_np.shape[1]}"
        if add_to_index:
            self.index.add(embeddings_np)
        D, I = self.index.search(embeddings_np, k=5)
        return torch.tensor(D, dtype=torch.float32), torch.tensor(I, dtype=torch.int64)
