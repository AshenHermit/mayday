from telegram.ext import ContextTypes, Application
from telegram import Update
import telegram
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
from character import Character
import random
import asyncio
import time
import os
import traceback
from deep_translator import GoogleTranslator

import nltk
nltk.download('punkt')


from_code = "ru"
to_code = "en"

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
enabled = True

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

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global enabled
    enabled = True
    await context.bot.send_message(chat_id=chat_id, text="[sys] activated")

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global enabled
    enabled = False
    await context.bot.send_message(chat_id=chat_id, text="[sys] deactivated")

async def callback_loop(context: ContextTypes.DEFAULT_TYPE):
    global in_thought
    global enabled
    if not enabled: return
    
    if random.uniform(0, 1.0) > max( 0.97 - in_thought*0.02, 0.85):
        await keep_talking(context)
        in_thought += 1

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global in_thought
    from_code = "ru"
    to_code = "en"
    text = update.message.text
    text = capitalize_sentences_nltk(text)
    translatedText = GoogleTranslator(source='ru', target='en').translate(text) 
    await read_and_answer(translatedText, context)
    in_thought = 0

application = Application.builder().token(os.environ["TELEGRAM_API_KEY"]).build()
job_queue = application.job_queue
job_loop = job_queue.run_repeating(callback_loop, interval=30, first=1)

echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)
application.add_handler(echo_handler)
application.add_handler(CommandHandler("keep", keep_command))
application.add_handler(CommandHandler("start", start_command))
application.add_handler(CommandHandler("stop", stop_command))


application.run_polling()

