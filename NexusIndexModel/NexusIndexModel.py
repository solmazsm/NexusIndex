;==========================================
; Title: NexusIndexModel
; Author: Solmaz Seyed Monir
;==========================================
class NexusIndexModel(nn.Module):
    def __init__(self, input_size):
        super(NexusIndexModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
