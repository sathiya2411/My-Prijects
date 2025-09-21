import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

class TweetDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    if df['label'].dtype == object:
        df['label'] = df['label'].map({'No': 0, 'Yes': 1}) # adjust for your labels
    return df

def get_dataloaders(csv_path, tokenizer, batch_size=16, max_length=128, test_size=0.2):
    df = load_dataset(csv_path)
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])
    train_dataset = TweetDataset(train_df, tokenizer, max_length)
    val_dataset = TweetDataset(val_df, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader

if __name__ == "__main__":
    from bert_mental_health_classifier import BertMentalHealthClassifier

    csv_path = "suicidewatch.csv"
    bert_model_name = 'bert-base-uncased'
    batch_size = 16
    num_labels = 2
    max_length = 128
    epochs = 2
    lr = 2e-5

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    train_loader, val_loader = get_dataloaders(csv_path, tokenizer, batch_size, max_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertMentalHealthClassifier(bert_model_name, num_labels).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} finished.")

    print("Training complete. Add evaluation and saving logic as needed.")