import streamlit as st
import numpy as np
import random
import json
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords

from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib import colormaps
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_valid_acc import accuracy_train, accuracy_valid
from nltk_utils import bag_of_words, tokenize, stem
from nn_model import NeuralNet

# Download NLTK resources if not already downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# pages = {
#     # "Tools": [
#     #     st.Page("nn_model.py", title="Model"),
#     # ],
#     "Chatbot": [
#         st.Page("imp_nlp.py", title="Text Pre-processing"),
#         st.Page("imp_nn.py", title="Training Model"),
#         # st.Page("eval.py", title="Evaluation"),
#     ],
# }

# pg = st.navigation(pages)
# pg.run()

optim_options = ["Adam", "Adamax", "RAdam"]

add_selectbox = st.sidebar.selectbox(
    "Select Optimizer",
    optim_options
)

st.write(
    f'<span style="font-size: 78px; line-height: 1">🐱</span>',
    unsafe_allow_html=True,
)

st.header("Neural Network Model", divider=True)
uploaded_file = st.file_uploader("Choose a JSON file", accept_multiple_files=False)
if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    st.write("Filename : ", uploaded_file.name)

    with open(uploaded_file.name, 'r') as f:
        intents = json.load(f)

    with st.expander("Tampilkan Isi File JSON"):
        st.json(intents, expanded=3)

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

    st.subheader("Data Preparation", divider=True)
    st.markdown("Memanfaatkan ***text pre-processing*** untuk mengolah dan mengubah data.")
    st.warning('Caution: rerun will make the dataset re-splitting', icon="⚠️")
    st.write("Patterns (Pola Pertanyaan) :", len(xy))
    st.write("Tags :", len(tags))
    with st.expander("Tampilkan Isi Data (Class)"):
        st.text(tags)
    st.write("Unique Words (Token) :", len(all_words))
    with st.expander("Tampilkan Isi Data (Token)"):
        st.text(all_words)
    
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

    # st.write('')
    # st.write('x_train :')
    # st.text(X_train[0])
    # st.write('y_train :', y_train[0])
    # st.text(df_y[0])
    # st.write('')

    # Train-Test-Validation split
    X_train2, X_restdata, y_train2, y_restdata = train_test_split(X_train, y_train, train_size=0.80, shuffle=True)
    X_test2, X_val2, y_test2, y_val2 = train_test_split(X_restdata, y_restdata, test_size=0.5, shuffle=True)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20)

    st.write('Data Latih : ', len(X_train2), '| Data Uji : ', len(X_test2), '| Data Validasi : ', len(X_val2))

    # Train Model Streamlit
    st.session_state.value = "Train Model"
    st.subheader(st.session_state.value, divider=True)

    # if st.button("Start Training"):
    #     st.session_state.value = "Start Training"
    #     st.rerun()
    # st.markdown("Memanfaatkan ***text pre-processing*** untuk mengolah dan mengubah data.")

    # Hyper-parameters 
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001
    input_size = len(X_train[0])
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


    if add_selectbox == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optim_selected = st.info("***Adam Optimizer*** is currently used.", icon="ℹ️")
    elif add_selectbox == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
        optim_selected = st.info("***Adamax Optimizer*** is currently used.", icon="ℹ️")
    elif add_selectbox == "RAdam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
        optim_selected = st.info("***RAdam Optimizer*** is currently used.", icon="ℹ️")

    optim_selected = optim_selected

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer
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
                log_dict['validation_loss_per_batch'].append(val_loss.item())
                val_losses.append(val_loss.item())
                # val_losses2 = loss.item() * words.size(1)

            #  computing accuracy
            val_accuracy = accuracy_valid(model, valid_loader)
            log_dict['validation_accuracy_per_epoch'].append(val_accuracy)

        train_losses = np.array(train_losses).mean()
        val_losses = np.array(val_losses).mean()

        log_dict['training_loss_per_epoch'].append(train_losses)
        log_dict['validation_loss_per_epoch'].append(val_losses)
        perplex_val2 = torch.exp(torch.tensor(val_losses)).item()

        print(f'Epoch [{epoch+1}/{num_epochs}],  training_loss: {round(train_losses, 4)}  training_accuracy: '+ f'{train_accuracy}  validation_loss: {round(val_losses, 4)} '+ f'validation_accuracy: {val_accuracy}')
        if (epoch+1) % 10 == 0:
            st.write(f'Epoch [{epoch+1}/{num_epochs}],  train_loss: {round(train_losses, 4)}  train_acc: '+ f'{train_accuracy}  valid_loss: {round(val_losses, 4)} '+ f'valid_acc: {val_accuracy}')
            st.write(f'Perplexity_val: {perplex_val2:.4f}')

    print(f'Final \t Train Loss: {round(train_losses, 4):.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {round(val_losses, 4):.4f}, Validation Accuracy: {val_accuracy:.4f}')
    st.write(f'Final: \t Train Loss: {round(train_losses, 4):.3f}, Train Accuracy: {train_accuracy:.3f}, Validation Loss: {round(val_losses, 4):.3f}, Validation Accuracy: {val_accuracy:.3f}')
    st.write(f'Perplexity: {perplex_val2:.4f}')

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

    # print(f'training complete. file saved to {FILE}')
    # st.write(f'training complete. file saved to ***{FILE}***')

    st.subheader("Visualization", divider=True)
    with st.expander("Grafik Plot Hasil Pelatihan Model"):
        fig, axs = plt.subplots(2, figsize=(8, 9), layout='constrained')
        axs[0].plot(log_dict['training_loss_per_epoch'], label='Training Loss')
        axs[0].plot(log_dict['validation_loss_per_epoch'], label='Validation Loss')
        axs[0].set_title("Model Loss")
        axs[0].set_ylabel("Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(log_dict['training_accuracy_per_epoch'], label='Training Accuracy')
        axs[1].plot(log_dict['validation_accuracy_per_epoch'], label='Validation Accuracy')
        axs[1].set_title("Model Accuracy")
        axs[1].set_ylabel("Accuracy")
        axs[1].set_xlabel("Epoch")
        axs[1].grid(True)
        axs[1].legend()

        st.pyplot(fig)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    x_test_tensor = torch.from_numpy(X_test2).to(device)
    y_test_tensor = torch.from_numpy(y_test2).to(device)

    with torch.no_grad(): # turn off Backpropagation
        # y_eval = prediction
        prediction = model.forward(x_test_tensor)
        #   loss_e = criterion(prediction, y_test_tensor)

    def convert_to_binary_matrix(list_of_numbers, max_value):
        binary_matrix = np.zeros((len(list_of_numbers), max_value))
        for i in range(len(list_of_numbers)):
            binary_matrix[i][list_of_numbers[i]] = 1
        return binary_matrix


    y_true = np.array(y_test2)
    list_of_numbers = y_true
    max_value = len(tags) # Jumlah tags
    binary_matrix = convert_to_binary_matrix(list_of_numbers, max_value)

    binary_matrix = np.array(binary_matrix)
    binary_matrix = binary_matrix.astype(int)
    # print(binary_matrix)

    cm = confusion_matrix(binary_matrix.argmax(axis=1), prediction.argmax(axis=1))

    class_name = tags
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels = class_name)

    y_predi = prediction.argmax(axis=1)
    y_predi = y_predi.numpy()

    y_true_cm = pd.Series(y_true, name="Actual")
    y_pred_cm = pd.Series(y_predi, name="Predicted")
    test_cm = pd.crosstab(y_true_cm, y_pred_cm)

    with st.expander("Confusion Matrix Plot"):
        fig_cm = plt.figure(figsize=(20,5))
        ax1 = plt.subplot(121)
        sn.heatmap(test_cm,annot=True,cmap="Blues",xticklabels=tags,yticklabels=tags)
        st.pyplot(fig_cm)

    with st.expander("Calculate : Accuracy / Precision / Recall / F1-Score"):
        # fig_cr = pd.DataFrame(classification_report(y_true_cm, y_pred_cm, target_names=class_name, output_dict=True)).T
        # st.write(fig_cr)

        accuracy_cm = metrics.accuracy_score(y_true_cm, y_pred_cm)
        precision_score_cm = metrics.precision_score(y_true_cm, y_pred_cm, average='macro')
        recall_score_cm = metrics.recall_score(y_true_cm, y_pred_cm, average='macro')
        f1_score_cm = metrics.f1_score(y_true_cm, y_pred_cm, average='macro')
        # specificity_cm = metrics.recall_score(y_true_cm, y_pred_cm, pos_label=0)

        # st.write(f'Accuracy : {round(accuracy_cm, 4):.3f}, Precision : {precision_score_cm:.3f}')
        # st.write(f'Recall : {round(recall_score_cm, 4):.3f}, F1-score: {f1_score_cm:.3f}')

        st.write("Accuracy :", round(accuracy_cm, 5), "Precision :", round(precision_score_cm, 5), "Recall :", round(recall_score_cm, 5), "F1 Score :", round(f1_score_cm, 5))
        # st.write("Accuracy :", round(accuracy_cm, 5), "Precision :", round(precision_score_cm, 5), "Recall :", round(recall_score_cm, 5), "F1 Score :", round(f1_score_cm, 5), "Specificity :", round(specificity_cm, 5))
        # st.write("Recall :", round(recall_score_cm, 5), "F1 Score :", round(f1_score_cm, 5))


    # # Chatbot test
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # with open(uploaded_file.name, 'r') as json_data:
    #     intents = json.load(json_data)

    # FILE = "data.pth"
    # data = torch.load(FILE)

    # input_size = data["input_size"]
    # hidden_size = data["hidden_size"]
    # output_size = data["output_size"]
    # all_words = data['all_words']
    # tags = data['tags']
    # model_state = data["model_state"]

    # model = NeuralNet(input_size, hidden_size, output_size).to(device)
    # model.load_state_dict(model_state)
    # model.eval()

    # bot_name = "NeuralBot"

    # # -- Code Website --
    # def get_response(msg):
    # # ------------------

    #     # sentence = tokenize(sentence)
    #     sentence = tokenize(msg)
    #     X = bag_of_words(sentence, all_words)
    #     X = X.reshape(1, X.shape[0])
    #     X = torch.from_numpy(X).to(device)

    #     output = model(X)
    #     _, predicted = torch.max(output, dim=1)

    #     tag = tags[predicted.item()]

    #     probs = torch.softmax(output, dim=1)
    #     prob = probs[0][predicted.item()]
    #     if prob.item() > 0.80:
    #         for intent in intents['intents']:
    #             if tag == intent["tag"]:
    #                 # -- Code Asli --
    #                 st.write(f"{random.choice(intent['responses'])} {intent['tag'], prob}")
    #                 st.write(f"{intent['tag'], probs}")
    #     else:
    #         st.write(f"{bot_name}: Aku tidak paham... {probs}")
    #         # -- Batas Code Asli --

    #                 # return random.choice(intent['responses'])
                
    #     # return "Aku tidak paham..."

    # st.title("Chatbot Test")

    # # Initialize chat history
    # if "messages" not in st.session_state:
    #     st.session_state.messages = []

    # # Display chat messages from history on app rerun
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

    # # React to user input
    # if prompt := st.chat_input("What is up?"):
    #     # Display user message in chat message container
    #     st.chat_message("user").markdown(prompt)
    #     # Add user message to chat history
    #     st.session_state.messages.append({"role": "user", "content": prompt})


    #     # Display assistant response in chat message container
    #     with st.chat_message("assistant"):
    #         resp = get_response(prompt)
    #     # Add assistant response to chat history
    #     st.session_state.messages.append({"role": "assistant", "content": resp})