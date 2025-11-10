import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Download NLTK resources if not already downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# --- Data Loading and Preprocessing ---
class TextDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.texts = [item['text'] for item in self.data]
        self.labels = [item['label'] for item in self.data]
        self.preprocess()

    def preprocess(self):
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        processed_texts = []
        for text in self.texts:
            tokens = word_tokenize(text.lower())
            tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
            processed_texts.append(" ".join(tokens))

        self.vectorizer = CountVectorizer()
        self.bow = self.vectorizer.fit_transform(processed_texts).toarray()
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.bow[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# --- Neural Network Model ---
class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- Streamlit App ---
st.title("Text Classification with PyTorch")

uploaded_file = st.file_uploader("Upload your JSON dataset", type="json")

if uploaded_file is not None:
    try:
        with open("temp.json", "wb") as f: #Temporary file to avoid issues
            f.write(uploaded_file.getbuffer())
        dataset = TextDataset("temp.json")
        input_dim = dataset.bow.shape[1]
        output_dim = len(set(dataset.labels))

        # Hyperparameters
        hidden_dim = st.slider("Hidden Dimension", 10, 200, 50)
        learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001)
        num_epochs = st.slider("Number of Epochs", 10, 100, 20)
        batch_size = st.slider("Batch Size", 1, 64, 32)

        # Data Loader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Model, Loss, and Optimizer
        model = TextClassifier(input_dim, hidden_dim, output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training Loop
        if st.button("Train Model"):
            with st.spinner("Training..."):
                for epoch in range(num_epochs):
                    for inputs, labels in dataloader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                st.success("Model trained!")
                st.write(f"Final Loss: {loss.item():.4f}")

        #Example prediction
        user_input = st.text_input("Enter text for prediction:")
        if st.button("Predict") and user_input and 'model' in locals(): #Check if model is trained
            processed_input = dataset.vectorizer.transform([user_input]).toarray()
            input_tensor = torch.tensor(processed_input, dtype=torch.float32)
            with torch.no_grad():
                prediction = model(input_tensor)
                predicted_class = torch.argmax(prediction).item()
                st.write(f"Predicted Class: {predicted_class}")


    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please upload a valid JSON file.")
    except Exception as e:
        st.error(f"An error occurred: {e}")