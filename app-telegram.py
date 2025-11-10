# from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS
# from chat import get_response

# # telegram
# from telebot.credentials import bot_token, bot_user_name, URL
# form telebot.mastermind import get_response

# app = Flask(__name__)
# CORS(app)

# # @app.get("/", methods=["GET"])
# # Use if not using standalone
# @app.get("/")
# def index_get():
#     return render_template("base.html")

# @app.post("/predict")
# def predict():
#     text = request.get_json().get("message")
#     # TODO: check if text is valid
#     response = get_response(text)
#     message = {"answer": response}
#     return jsonify(message)

# # telegram
# @app.route('/setwebhook', methods=['GET', 'POST'])
# def set_webhook():
#     s = bot.setWebhook('{URL}{HOOK}'.format(URL=URL, HOOK=TOKEN))
#     if s:
#         return "webhook setup ok"
#     else:
#         return "webhook setup failed"

# @app.route('/')
# def index():
#     return '.'

# if __name__ == "__main__":
#     app.run(debug=True)