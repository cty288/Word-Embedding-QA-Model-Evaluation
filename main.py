from RNNModel import *
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import json
from tfidf_generator import TFIDFGenerator
import gensim.downloader as api
from word2vec_generator import Word2VecGenerator
import torch.nn.functional as F
from sklearn.metrics import f1_score

#word2vec_model = api.load("word2vec-google-news-300")

def read_json_data(file_path, limit=None):
    with open(file_path, 'r', encoding= "utf-8") as f:
        json_str = f.read()
    json_obj = json.loads(json_str)
    return json_obj[:limit]

#评估函数

def evaluate_with_tolerance(true_start_positions, predicted_start_positions, tolerance=1):
    within_tolerance = 0
    for true_pos, pred_pos in zip(true_start_positions, predicted_start_positions):
        if abs(true_pos - pred_pos) <= tolerance:
            within_tolerance += 1
    return within_tolerance / len(true_start_positions)

def exact_match(y_true, y_pred):
    return sum([1 if pred == true else 0 for pred, true in zip(y_pred, y_true)]) / len(y_true)


def calculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro') 

 
#读Json，dev懒得读了
train_set = read_json_data("train_set_splited.json")
test_set = read_json_data("test_set_splited.json")


#三个model调用获得vector
train_articles, train_questions, train_answer_starts, train_answers = TFIDFGenerator(train_set).vectorize()
test_articles, test_questions, test_answer_starts, train_answers = TFIDFGenerator(test_set).vectorize()


#不懂，GPT写的，应该是转换成tensor
train_answer_starts = torch.tensor(train_answer_starts, dtype=torch.long)
test_answer_starts = torch.tensor(test_answer_starts, dtype=torch.long)

#不懂，GPT写的，应该和上面差不多
train_dataloader = DataLoader(TensorDataset(train_articles, train_questions, train_answer_starts), batch_size=32)
test_dataloader = DataLoader(TensorDataset(test_articles, test_questions, test_answer_starts), batch_size=32)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#RNN参数
input_size = 1200
hidden_size = 256
num_layers = 3
model = RNNModel(input_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
no_answer_criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.to(device)

# Train the model
num_epochs = 10

#开跑！
train(model, train_dataloader, criterion, no_answer_criterion, optimizer, device, num_epochs)

# Test the model
test_loss, accuracy_no_answer, accuracy_answer_start, true_start_positions, predicted_start_positions = test(model, test_dataloader, criterion, no_answer_criterion, device)
print(f"Test Loss: {test_loss:.4f}, No Answer Accuracy: {accuracy_no_answer:.4f}, Answer Start Accuracy: {accuracy_answer_start:.4f}")

accuracy_within_tolerance = evaluate_with_tolerance(true_start_positions, predicted_start_positions, tolerance=5)
print(f"Accuracy within tolerance: {accuracy_within_tolerance:.4f}")



exact_match_score = exact_match(true_start_positions, predicted_start_positions)
f1_score_result = calculate_f1(true_start_positions, predicted_start_positions)
print(f"Exact Match: {exact_match_score:.4f}, F1 Score: {f1_score_result:.4f}")