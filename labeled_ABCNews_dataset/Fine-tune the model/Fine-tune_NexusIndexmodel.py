
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.train()
for epoch in range(NUM_EPOCHS):
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

