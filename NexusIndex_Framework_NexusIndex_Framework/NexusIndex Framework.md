```python

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


X_val_tensor = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)


train_losses = []
val_losses = []
epochs = []


num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    
    train_losses.append(loss.item())
    
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses.append(val_loss.item())

    
    epochs.append(epoch + 1)

    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")


plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('(NexusIndex Framework)', fontsize=14, fontweight='bold')
plt.legend()
plt.annotate('NexusIndex Framework', xy=(0.5, 0.5), xycoords='axes fraction', fontsize=12,
             xytext=(0.5, 0.7), textcoords='axes fraction',
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()


model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    predictions = (test_outputs >= 0.5).float()
    accuracy = accuracy_score(y_test_tensor.numpy(), predictions.numpy())
    print(f"Test Loss: {test_loss.item():.4f}, Test Accuracy: {accuracy:.4f}")
plt.savefig('training_validation_loss_plot.pdf', bbox_inches='tight') 
```

    Epoch 1/1000, Train Loss: 0.0161, Val Loss: 0.0101
    Epoch 11/1000, Train Loss: 0.0130, Val Loss: 0.0083
    Epoch 21/1000, Train Loss: 0.0107, Val Loss: 0.0069
    Epoch 31/1000, Train Loss: 0.0090, Val Loss: 0.0059
    Epoch 41/1000, Train Loss: 0.0077, Val Loss: 0.0051
    Epoch 51/1000, Train Loss: 0.0066, Val Loss: 0.0044
    Epoch 61/1000, Train Loss: 0.0058, Val Loss: 0.0039
    Epoch 71/1000, Train Loss: 0.0051, Val Loss: 0.0035
    Epoch 81/1000, Train Loss: 0.0045, Val Loss: 0.0031
    Epoch 91/1000, Train Loss: 0.0041, Val Loss: 0.0028
    Epoch 101/1000, Train Loss: 0.0036, Val Loss: 0.0026
    Epoch 111/1000, Train Loss: 0.0033, Val Loss: 0.0023
    Epoch 121/1000, Train Loss: 0.0030, Val Loss: 0.0022
    Epoch 131/1000, Train Loss: 0.0027, Val Loss: 0.0020
    Epoch 141/1000, Train Loss: 0.0025, Val Loss: 0.0018
    Epoch 151/1000, Train Loss: 0.0023, Val Loss: 0.0017
    Epoch 161/1000, Train Loss: 0.0021, Val Loss: 0.0016
    Epoch 171/1000, Train Loss: 0.0020, Val Loss: 0.0015
    Epoch 181/1000, Train Loss: 0.0018, Val Loss: 0.0014
    Epoch 191/1000, Train Loss: 0.0017, Val Loss: 0.0013
    Epoch 201/1000, Train Loss: 0.0016, Val Loss: 0.0012
    Epoch 211/1000, Train Loss: 0.0015, Val Loss: 0.0011
    Epoch 221/1000, Train Loss: 0.0014, Val Loss: 0.0011
    Epoch 231/1000, Train Loss: 0.0013, Val Loss: 0.0010
    Epoch 241/1000, Train Loss: 0.0012, Val Loss: 0.0010
    Epoch 251/1000, Train Loss: 0.0012, Val Loss: 0.0009
    Epoch 261/1000, Train Loss: 0.0011, Val Loss: 0.0009
    Epoch 271/1000, Train Loss: 0.0010, Val Loss: 0.0008
    Epoch 281/1000, Train Loss: 0.0010, Val Loss: 0.0008
    Epoch 291/1000, Train Loss: 0.0009, Val Loss: 0.0007
    Epoch 301/1000, Train Loss: 0.0009, Val Loss: 0.0007
    Epoch 311/1000, Train Loss: 0.0008, Val Loss: 0.0007
    Epoch 321/1000, Train Loss: 0.0008, Val Loss: 0.0007
    Epoch 331/1000, Train Loss: 0.0008, Val Loss: 0.0006
    Epoch 341/1000, Train Loss: 0.0007, Val Loss: 0.0006
    Epoch 351/1000, Train Loss: 0.0007, Val Loss: 0.0006
    Epoch 361/1000, Train Loss: 0.0007, Val Loss: 0.0005
    Epoch 371/1000, Train Loss: 0.0006, Val Loss: 0.0005
    Epoch 381/1000, Train Loss: 0.0006, Val Loss: 0.0005
    Epoch 391/1000, Train Loss: 0.0006, Val Loss: 0.0005
    Epoch 401/1000, Train Loss: 0.0006, Val Loss: 0.0005
    Epoch 411/1000, Train Loss: 0.0005, Val Loss: 0.0004
    Epoch 421/1000, Train Loss: 0.0005, Val Loss: 0.0004
    Epoch 431/1000, Train Loss: 0.0005, Val Loss: 0.0004
    Epoch 441/1000, Train Loss: 0.0005, Val Loss: 0.0004
    Epoch 451/1000, Train Loss: 0.0005, Val Loss: 0.0004
    Epoch 461/1000, Train Loss: 0.0004, Val Loss: 0.0004
    Epoch 471/1000, Train Loss: 0.0004, Val Loss: 0.0004
    Epoch 481/1000, Train Loss: 0.0004, Val Loss: 0.0004
    Epoch 491/1000, Train Loss: 0.0004, Val Loss: 0.0003
    Epoch 501/1000, Train Loss: 0.0004, Val Loss: 0.0003
    Epoch 511/1000, Train Loss: 0.0004, Val Loss: 0.0003
    Epoch 521/1000, Train Loss: 0.0004, Val Loss: 0.0003
    Epoch 531/1000, Train Loss: 0.0003, Val Loss: 0.0003
    Epoch 541/1000, Train Loss: 0.0003, Val Loss: 0.0003
    Epoch 551/1000, Train Loss: 0.0003, Val Loss: 0.0003
    Epoch 561/1000, Train Loss: 0.0003, Val Loss: 0.0003
    Epoch 571/1000, Train Loss: 0.0003, Val Loss: 0.0003
    Epoch 581/1000, Train Loss: 0.0003, Val Loss: 0.0003
    Epoch 591/1000, Train Loss: 0.0003, Val Loss: 0.0003
    Epoch 601/1000, Train Loss: 0.0003, Val Loss: 0.0002
    Epoch 611/1000, Train Loss: 0.0003, Val Loss: 0.0002
    Epoch 621/1000, Train Loss: 0.0003, Val Loss: 0.0002
    Epoch 631/1000, Train Loss: 0.0003, Val Loss: 0.0002
    Epoch 641/1000, Train Loss: 0.0002, Val Loss: 0.0002
    Epoch 651/1000, Train Loss: 0.0002, Val Loss: 0.0002
    Epoch 661/1000, Train Loss: 0.0002, Val Loss: 0.0002
    Epoch 671/1000, Train Loss: 0.0002, Val Loss: 0.0002
    Epoch 681/1000, Train Loss: 0.0002, Val Loss: 0.0002
    Epoch 691/1000, Train Loss: 0.0002, Val Loss: 0.0002
    Epoch 701/1000, Train Loss: 0.0002, Val Loss: 0.0002
    Epoch 711/1000, Train Loss: 0.0002, Val Loss: 0.0002
    Epoch 721/1000, Train Loss: 0.0002, Val Loss: 0.0002
    Epoch 731/1000, Train Loss: 0.0002, Val Loss: 0.0002
    Epoch 741/1000, Train Loss: 0.0002, Val Loss: 0.0002
    Epoch 751/1000, Train Loss: 0.0002, Val Loss: 0.0002
    Epoch 761/1000, Train Loss: 0.0002, Val Loss: 0.0002
    Epoch 771/1000, Train Loss: 0.0002, Val Loss: 0.0002
    Epoch 781/1000, Train Loss: 0.0002, Val Loss: 0.0002
    Epoch 791/1000, Train Loss: 0.0002, Val Loss: 0.0001
    Epoch 801/1000, Train Loss: 0.0002, Val Loss: 0.0001
    Epoch 811/1000, Train Loss: 0.0002, Val Loss: 0.0001
    Epoch 821/1000, Train Loss: 0.0002, Val Loss: 0.0001
    Epoch 831/1000, Train Loss: 0.0002, Val Loss: 0.0001
    Epoch 841/1000, Train Loss: 0.0001, Val Loss: 0.0001
    Epoch 851/1000, Train Loss: 0.0001, Val Loss: 0.0001
    Epoch 861/1000, Train Loss: 0.0001, Val Loss: 0.0001
    Epoch 871/1000, Train Loss: 0.0001, Val Loss: 0.0001
    Epoch 881/1000, Train Loss: 0.0001, Val Loss: 0.0001
    Epoch 891/1000, Train Loss: 0.0001, Val Loss: 0.0001
    Epoch 901/1000, Train Loss: 0.0001, Val Loss: 0.0001
    Epoch 911/1000, Train Loss: 0.0001, Val Loss: 0.0001
    Epoch 921/1000, Train Loss: 0.0001, Val Loss: 0.0001
    Epoch 931/1000, Train Loss: 0.0001, Val Loss: 0.0001
    Epoch 941/1000, Train Loss: 0.0001, Val Loss: 0.0001
    Epoch 951/1000, Train Loss: 0.0001, Val Loss: 0.0001
    Epoch 961/1000, Train Loss: 0.0001, Val Loss: 0.0001
    Epoch 971/1000, Train Loss: 0.0001, Val Loss: 0.0001
    Epoch 981/1000, Train Loss: 0.0001, Val Loss: 0.0001
    Epoch 991/1000, Train Loss: 0.0001, Val Loss: 0.0001
    


    
![png](output_0_1.png)
    


    Test Loss: 0.0001, Test Accuracy: 1.0000
    


    <Figure size 640x480 with 0 Axes>



```python

```
