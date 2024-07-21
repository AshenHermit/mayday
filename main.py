from telegram.ext import ContextTypes, Application
from telegram import Update
import telegram
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
from character import Character
import random
import asyncio
import time
import argostranslate.translate
import argostranslate.package
import os
import traceback

import nltk
nltk.download('punkt')


from_code = "ru"
to_code = "en"

# Download and install Argos Translate package
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(
        lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
    )
)
argostranslate.package.install_from_path(package_to_install.download())

def capitalize_sentences_nltk(text):
    # Токенизируем текст на предложения
    sentences = nltk.tokenize.sent_tokenize(text)
    
    # Преобразуем первую букву каждого предложения в заглавную
    capitalized_sentences = [sentence.capitalize() for sentence in sentences]
    
    # Объединяем предложения обратно в текст
    return ' '.join(capitalized_sentences)

chat_id = os.environ["TELEGRAM_CHAT_ID"]

char = Character()

in_thought = 0

random.seed = time.time()

async def wait_for_read(text):
    if not text: return
    await asyncio.sleep(random.uniform(len(text) * 0.01, len(text) * 0.02))
async def wait_for_answer(text):
    if not text: return
    await asyncio.sleep(random.uniform(len(text) * 0.02, len(text) * 0.05))

async def keep_talking(context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=chat_id, action=telegram.constants.ChatAction.TYPING)
    try:
        answer = char.chat()
    except:
        answer = f"error: {traceback.format_exc()}"
    if answer: await context.bot.send_message(chat_id=chat_id, text=answer)
    
async def read_and_answer(message, context: ContextTypes.DEFAULT_TYPE):
    await wait_for_read(message)
    await context.bot.send_chat_action(chat_id=chat_id, action=telegram.constants.ChatAction.TYPING)
    try:
        answer = char.chat(message)
    except:
        answer = f"error: {traceback.format_exc()}"
    if answer: await context.bot.send_message(chat_id=chat_id, text=answer)

async def keep_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await keep_talking(context)

async def callback_loop(context: ContextTypes.DEFAULT_TYPE):
    global in_thought
    if random.uniform(0, 1.0) > max( 0.97 - in_thought*0.02, 0.85):
        await keep_talking(context)
        in_thought += 1

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global in_thought
    from_code = "ru"
    to_code = "en"
    text = update.message.text
    text = capitalize_sentences_nltk(text)
    translatedText = argostranslate.translate.translate(text, from_code, to_code)
    await read_and_answer(translatedText, context)
    in_thought = 0

application = Application.builder().token(os.environ["TELEGRAM_API_KEY"]).build()
job_queue = application.job_queue
job_loop = job_queue.run_repeating(callback_loop, interval=30, first=1)

echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)
application.add_handler(echo_handler)
application.add_handler(CommandHandler("keep", keep_command))


application.run_polling()

