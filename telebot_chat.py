import random
import json
import torch
from numpy import savetxt
# import nltk
# nltk.download('stopwords')

# from nltk.corpus import stopwords
from nn_model import NeuralNet
from nltk_utils import bag_of_words, tokenize, stem


# stopWords = list(set(stopwords.words('english')))
bot_name = "NeuralNetBot"

# -- Code --
def get_response(msg):
# ------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE)

    # failed to answer questions history (temporary)
    fail_msg = []
    old_data = open("fail_msg.txt", "r").read().split('\n')
    fail_msg.extend(old_data)    
    # f = open(fail_msg.txt”, “x”)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # sentence = tokenize(sentence)
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # -- Code Asli --
    #             print(f"{bot_name}: {random.choice(intent['responses'])} {intent['tag'], prob}")
    #             print(f"{intent['tag'], probs}")
    # else:
    #     print(f"{bot_name}: I do not understand... {probs}")
        # -- Batas Code Asli --
                # print(f"{intent['tag'], prob}")
                return random.choice(intent['responses'])
            
    fail_msg.append(msg)
    with open("fail_msg.txt", "w") as out:
        out.write('\n'.join(fail_msg))
    return "Aku tidak paham..."

# if __name__ == "__main__":
#     print("Let's chat! (type 'quit' to exit)")
#     while True:
#         # sentence = "do you use credit cards?"
#         sentence = input("You: ")
#         if sentence == "quit":
#             break

#         resp = get_response(sentence)
#         print(resp)