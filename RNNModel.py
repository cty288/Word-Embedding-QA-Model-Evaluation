import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 4, 1) # Update: hidden_size * 2 => hidden_size * 4
        self.fc_no_answer = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, article_vectors, question_vectors):
        hidden_article, _ = self.rnn(article_vectors.unsqueeze(1))
        hidden_question, _ = self.rnn(question_vectors.unsqueeze(1))

        hidden_question_expanded = hidden_question.repeat(1, article_vectors.size(1), 1)
        hidden_article_expanded = hidden_article.repeat(1, article_vectors.size(1), 1) # Update: expand hidden_article
        hidden_combined = torch.cat([hidden_article_expanded, hidden_question_expanded], dim=2)

        answer_start_prob = self.fc(hidden_combined).squeeze(2)

        no_answer_prob = self.sigmoid(self.fc_no_answer(hidden_question))

        return answer_start_prob, no_answer_prob
    

def one_hot(tensor, num_classes):
    one_hot_tensor = torch.zeros(tensor.size(0), num_classes, device=tensor.device)
    one_hot_tensor.scatter_(1, tensor.unsqueeze(1), 1)
    return one_hot_tensor

def train(model, dataloader, answer_criterion, no_answer_criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for article, question, answer_start in dataloader:
            optimizer.zero_grad()
            answer_start_prob, no_answer_prob = model(article.to(device), question.to(device))

            no_answer_target = (answer_start == -1).float().unsqueeze(1).to(device)  # Add a dimension to match no_answer_prob's shape
            answer_start = answer_start.clamp(min=0, max=article.size(1) - 1)  # Clamp answer_start to avoid out-of-bounds error
            
            
            answer_start_one_hot = one_hot(answer_start.squeeze().to(device), article.size(1))
            loss = F.mse_loss(answer_start_prob, answer_start_one_hot) + no_answer_criterion(no_answer_prob.squeeze(), no_answer_target.squeeze())

            #loss = F.mse_loss(answer_start_prob, answer_start.squeeze().to(device)) + no_answer_criterion(no_answer_prob.squeeze(), no_answer_target.squeeze())

            #loss = answer_criterion(answer_start_prob, answer_start.squeeze().to(device)) + no_answer_criterion(no_answer_prob.squeeze(), no_answer_target.squeeze())
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

        
      
        

def test(model, dataloader, answer_criterion, no_answer_criterion, device):
    model.eval()
    test_loss = 0
    correct_answer_start = 0
    correct_no_answer = 0
    total = 0
    
    true_start_positions = []
    predicted_start_positions = []

    with torch.no_grad():
        for article, question, answer_start in dataloader:
            total += article.size(0)

            answer_start_prob, no_answer_prob = model(article.to(device), question.to(device))

            no_answer_target = (answer_start == -1).float().unsqueeze(1).to(device)  # Add a dimension to match no_answer_prob's shape
            answer_start = answer_start.clamp(min=0, max=article.size(1) - 1)  # Clamp answer_start to avoid out-of-bounds error
            #answer_loss = answer_criterion(answer_start_prob, answer_start.squeeze().to(device))
            
            answer_start_one_hot = one_hot(answer_start.squeeze().to(device), article.size(1))
            answer_loss = F.mse_loss(answer_start_prob, answer_start_one_hot)
            
            #answer_loss = F.mse_loss(answer_start_prob, answer_start.squeeze().to(device))
            no_answer_loss = no_answer_criterion(no_answer_prob.squeeze(), no_answer_target.squeeze())
            loss = answer_loss + no_answer_loss
            test_loss += loss.item()

            # Modify accuracy calculation to handle the special case
            pred_start_pos = answer_start_prob.argmax(dim=1)
            pred_no_answer = (no_answer_prob > 0.5).float()
            correct_no_answer += (pred_no_answer == no_answer_target).sum().item()

            # Only consider answer_start predictions for samples with actual answers
            has_answer = answer_start != 0
            correct_answer_start += (pred_start_pos[has_answer] == answer_start[has_answer]).sum().item()
            
            true_start_positions.extend(answer_start.tolist())
            predicted_start_positions.extend(pred_start_pos.tolist())


    test_loss /= total
    accuracy_no_answer = correct_no_answer / total
    accuracy_answer_start = correct_answer_start / has_answer.sum().item()
    


    return test_loss, accuracy_no_answer, accuracy_answer_start, true_start_positions, predicted_start_positions



