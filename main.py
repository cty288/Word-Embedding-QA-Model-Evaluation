from QAModel import *
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import json
from tfidf_generator import TFIDFGenerator

def read_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

train_set = read_json_data("train_set_splited.json")
test_set = read_json_data("test_set_splited.json")


train_articles, train_questions, train_answer_starts = TFIDFGenerator(train_set).vectorize()
test_articles, test_questions, test_answer_starts = TFIDFGenerator(test_set).vectorize()



train_answer_starts = torch.tensor(train_answer_starts, dtype=torch.long)
test_answer_starts = torch.tensor(test_answer_starts, dtype=torch.long)


train_dataloader = DataLoader(TensorDataset(train_articles, train_questions, train_answer_starts), batch_size=32)
test_dataloader = DataLoader(TensorDataset(test_articles, test_questions, test_answer_starts), batch_size=32)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 300
hidden_size = 128
num_layers = 1
model = RNNModel(input_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
no_answer_criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.to(device)

# Train the model
num_epochs = 10
train(model, train_dataloader, criterion, no_answer_criterion, optimizer, device, num_epochs)

# Test the model
test_loss, accuracy_no_answer, accuracy_answer_start = test(model, test_dataloader, criterion, no_answer_criterion, device)
print(f"Test Loss: {test_loss:.4f}, No Answer Accuracy: {accuracy_no_answer:.4f}, Answer Start Accuracy: {accuracy_answer_start:.4f}")