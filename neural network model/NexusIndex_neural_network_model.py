
max_sequence_length = 100  
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
X = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(X, maxlen=max_sequence_length)
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100 
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    LSTM(64),  # Example LSTM layer with 64 units
    Dense(1, activation='sigmoid')
])


