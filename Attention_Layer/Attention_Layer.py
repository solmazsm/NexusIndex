# Attention Layer Proposed
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim, 1))
    
    def forward(self, x):
        attention_scores = torch.matmul(x, self.attention_weights).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=0).unsqueeze(-1)
        weighted_sum = torch.sum(x * attention_weights, dim=0)
        return weighted_sum
