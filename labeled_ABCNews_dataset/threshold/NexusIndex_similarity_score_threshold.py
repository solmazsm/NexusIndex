
threshold = 0.5

df['label'] = (df['Similarity (GPT-2)'] >= threshold).astype(int)

