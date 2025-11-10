import telebot_chat as nn
from telebot_chat import get_response

import logging
import os
from telegram import ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
# from telegram import InlineKeyboardButton, InlineKeyboardMarkup
# from telegram import ParseMode


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.info('Starting Bot...')

logger = logging.getLogger(__name__)

API_KEY = "6778822834:AAEsnemAro7l7tl0SBWEbdzhz8oGOJ_UDwo"

TELEGRAM_API_KEY: str = os.getenv('TELEGRAM_TOKEN')

# Commands
def start_command(update, context):
    # first_nem = message.from_user.first_name
    # first_nem = update.message.first_name
    buttons = [["/menu", "/help"]]
    keyboard1 = ReplyKeyboardMarkup(buttons, resize_keyboard=True, one_time_keyboard=True)
    update.message.reply_text("Halo! senang bertemu denganmu. Ada yang bisa saya bantu?", reply_markup=keyboard1)

def help_command(update, context):
    # first_nem = message.from_user.first_name
    # first_nem = update.message.first_name
    update.message.reply_text("Ini adalah bot seputar info Butik, silahkan ajukan pertanyaan seputar Butik. \n\n *Catatan disarankan untuk menggunakan ejaan bahasa Indonesia yang baik dan benar.")

def menu_command(update, context):
    update.message.reply_text(
    """
    /start -> Welcome to the channel \n/help -> How to use this bot \n/menu -> This praticular message
    """
    )

def handle_message(update, context):
    msg = str(update.message.text).lower()
    logging.info(f'User ({update.message.chat.id}) says: {msg}')

    # Bot response
    # update.message.reply_text(get_response)
    update.message.reply_text(get_response(msg))
    # update.message.reply_text(msg)

def error (update, context):
    # Logs errors
    logging.error(f'Update {update} caused error {context.error}')

# if __name__ == '__main__':
def main():
    updater = Updater(TELEGRAM_API_KEY, use_context=True)
    dp = updater.dispatcher

    # Commands
    dp.add_handler(CommandHandler('start', start_command))
    dp.add_handler(CommandHandler('help', help_command))
    dp.add_handler(CommandHandler('menu', menu_command))

    # Messages
    dp.add_handler(MessageHandler(Filters.text, handle_message))

    # Log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

