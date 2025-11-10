import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy_train(model, train_loader):
  # set model to evaluation mode
  model.eval()
  total_correct = 0
  total_instances = 0
  for (words, labels) in train_loader:
    # sending data to device
    words = words.to(device)
    labels = labels.to(dtype=torch.long).to(device)

    predictions = torch.argmax(model(words), dim=1)
    correct_predictions = sum(predictions==labels).item()
    total_correct+=correct_predictions
    total_instances+=len(words)
  return round(total_correct/total_instances, 3)

def accuracy_valid(model, valid_loader):
  # set model to evaluation mode
  model.eval()
  total_correct = 0
  total_instances = 0
  for (words, labels) in valid_loader:
    # sending data to device
    words = words.to(device)
    labels = labels.to(dtype=torch.long).to(device)

    predictions = torch.argmax(model(words), dim=1)
    correct_predictions = sum(predictions==labels).item()
    total_correct+=correct_predictions
    total_instances+=len(words)
  return round(total_correct/total_instances, 3)