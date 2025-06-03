from preprocess_text import build_vocab
import torch
from torch.utils.data import Dataset, DataLoader
from config import num_classes, BATCH_SIZE, FFN_HID_DIM, NUM_LAYERS, device, NHEAD, MAX_LEN, EMBED_DIM, EPOCHS, LR
from sklearn.model_selection import train_test_split
from dataset import CommentDataset
import pandas as pd
import torch.nn as nn
from transformer import TextTransformer
from tqdm import tqdm


def main():
    train_df = pd.read_csv('dataset/train.csv')[:1000]

    vocab = build_vocab(train_df['text'])
    VOCAB_SIZE = len(vocab)
    idx_to_token = {i: token for token, i in vocab.items()}


    dataset = CommentDataset(train_df, vocab, MAX_LEN)
    train_data, val_data = train_test_split(dataset, test_size=0.1)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = TextTransformer(
        num_classes=num_classes,
        vocab_size=VOCAB_SIZE,
        emb_size=EMBED_DIM,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=FFN_HID_DIM
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, src_key_padding_mask=(input_ids == 0))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, src_key_padding_mask=(input_ids == 0))
                predicted = torch.argmax(outputs, dim=1)

                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total if val_total > 0 else 0

        print(f"Epoch {epoch+1}, "
            f"Loss: {total_loss/len(train_loader):.4f}, "
            f"Train Acc: {train_acc:.2f}%, "
            f"Val Acc: {val_acc:.2f}%")
            
if __name__ == '__main__':
    main()