# NexusIndexModel proposed
;==========================================
; Title: A Novel NexusIndexModel for Fake News Detection
; Author: Solmaz Seyed Monir
;==========================================
class NexusIndexModel(nn.Module):
    def __init__(self, input_dim):
        super(NexusIndexModel, self).__init__()
        self.attention = AttentionLayer(input_dim)
        self.faiss_layer = FAISSLayer(input_dim)
        self.fc1 = nn.Linear(5, 128)  # Increased hidden units for more complexity
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)  # Dropout layer
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
    
    def forward(self, x, add_to_index=False):
        batch_size = x.size(0)
        x = self.attention(x)
        x = x.unsqueeze(0)
        D, I = self.faiss_layer(x, add_to_index)
        
        if D.dim() == 2 and D.size(0) == 1:
            D = D.repeat(batch_size, 1)
        
        D = D.view(batch_size, -1)
        
        x = F.relu(self.fc1(D))
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
