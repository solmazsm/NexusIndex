X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


input_size = X_train_scaled.shape[1]
model = NexusIndexModel(input_size)


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001
