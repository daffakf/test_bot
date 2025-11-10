import numpy as np
import random
import json

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_valid_acc import accuracy_train, accuracy_valid
from nltk_utils import bag_of_words, tokenize, stem
from nn_model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!', ',', ':', ';', '(', ')', '[', ']', '&', '..', '...']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# stopwords
list_stopwords = set(stopwords.words('indonesian'))
# remove stopword from token list
all_words = [w for w in all_words if w not in list_stopwords]

# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print('')
print("Patterns :", len(xy))
print("Tags :", len(tags))
print("Unique Words (Token) :", len(all_words))
print('')
print("Patterns (Pola Pertanyaan) :", len(xy))
print('')
print("Tags :", len(tags))
print(tags)
print('')
print("Unique Words (Token) :", len(all_words))
print(all_words)

output_empty = [0] * len(tags)
df_y = []

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    # label = bag_of_words(tag, tags)
    y_train.append(label)

    output_row = list(output_empty)
    output_row[tags.index(tag)] = 1
    df_y.append(output_row)

X_train = np.array(X_train)
y_train = np.array(y_train)

print('')
print('x_train :', X_train[0])
print('')
print('y_train :', y_train[0], '->', df_y[0])
print('')

# Train-Test-Validation split
# Testing
# X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.10)
# X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train2, y_train2, test_size=0.10)

X_train2, X_restdata, y_train2, y_restdata = train_test_split(X_train, y_train, train_size=0.80, shuffle=True)
X_test2, X_val2, y_test2, y_val2 = train_test_split(X_restdata, y_restdata, test_size=0.5, shuffle=True)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20)

# Hyper-parameters 
num_epochs = 100
batch_size = 32
learning_rate = 0.001
input_size = len(X_train2[0])
hidden_size = 128
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train2)
        self.x_data = X_train2
        self.y_data = y_train2

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

class ChatValid(Dataset):

    def __init__(self, X_val2, y_val2):
        self.n_samples = len(X_val2)
        self.x_data = X_val2
        self.y_data = y_val2

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

train_data = ChatDataset()

# validation_data = ChatValid()
validation_data = ChatValid(torch.from_numpy(X_val2), torch.from_numpy(y_val2))

train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

valid_loader = DataLoader(dataset=validation_data, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
min_valid_loss = np.inf
# Adam, Adamax, AdamW, RAdam, NAdam, RMSprop

#  creating log
log_dict = {
  'training_loss_per_batch': [],
  'validation_loss_per_batch': [],
  'training_loss_per_epoch': [],
  'validation_loss_per_epoch': [],
  'training_accuracy_per_epoch': [],
  'validation_accuracy_per_epoch': []
}

# Train the model with validation
for epoch in range(num_epochs):
  train_losses = []
  # train_losses2 = 0.0
  # Training or model.train()
  for (words, labels) in train_loader:
    # sending data to device
    words = words.to(device)
    labels = labels.to(dtype=torch.long).to(device)

    # Forward pass
    outputs = model(words)

    # computing loss
    loss = criterion(outputs, labels)
    log_dict['training_loss_per_batch'].append(loss.item())
    train_losses.append(loss.item())

    # Backward and optimize
    optimizer.zero_grad() # zeroing optimizer gradients
    loss.backward() # computing gradients
    optimizer.step() # updating weights
    # train_losses2 += loss.item()

  with torch.no_grad():
    #  computing training accuracy
    train_accuracy = accuracy_train(model, train_loader)
    log_dict['training_accuracy_per_epoch'].append(train_accuracy)

    # Accuracy
    # predictions = torch.argmax(outputs, dim=1)
    # accuracy = (predictions == labels).float().mean()

  # Validation
  val_losses = []
  perplex_val = []
  total_loss_perp = 0
  model.eval()
  with torch.no_grad():
    for (words, labels) in valid_loader:
      #  sending data to device
      words = words.to(device)
      labels = labels.to(dtype=torch.long).to(device)

      #  making predictions
      outputs = model(words)

      #  computing loss
      val_loss = criterion(outputs, labels)
      total_loss_perp += val_loss.item()
      log_dict['validation_loss_per_batch'].append(val_loss.item())
      val_losses.append(val_loss.item())
      # val_losses2 = loss.item() * words.size(1)
    avge_loss = total_loss_perp / len(valid_loader)
    perplex_val = torch.exp(torch.tensor(avge_loss))

    #  computing accuracy
    val_accuracy = accuracy_valid(model, valid_loader)
    log_dict['validation_accuracy_per_epoch'].append(val_accuracy)

  train_losses = np.array(train_losses).mean()
  val_losses = np.array(val_losses).mean()

  log_dict['training_loss_per_epoch'].append(train_losses)
  log_dict['validation_loss_per_epoch'].append(val_losses)
  perplex_val2 = torch.exp(torch.tensor(val_losses)).item()

  print(f'Epoch [{epoch+1}/{num_epochs}],  training_loss: {round(train_losses, 4)}  training_accuracy: '+ f'{train_accuracy}  validation_loss: {round(val_losses, 4)} '+ f'validation_accuracy: {val_accuracy}')
  print(f'Perplexity_val: {perplex_val2:.4f}')
  # print(f'Perplexity_val: {perplex_val:.4f}, Perplexity_val2: {perplex_val2:.4f}')

print(f'Final \t Train Loss: {round(train_losses, 4):.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {round(val_losses, 4):.4f}, Validation Accuracy: {val_accuracy:.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')