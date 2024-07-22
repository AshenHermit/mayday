from groq import Groq
from messages_history import MessagesHistory, ChatInstruction, ThoughtInstruction, SummaryInstruction
from datetime import datetime
import re
import httpx
import os
from pathlib import Path

class LLM():
    def __init__(self) -> None:
        self.groq_token = os.environ["GROQ_TOKEN"]
        self.client = Groq(api_key=self.groq_token)

    def generate(self, messages, temperature=0.87):
        completion = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        choice = completion.choices[0]
        return {
            'role': choice.message.role,
            'content': choice.message.content,
        }
    
class OllamaLLM():
    def __init__(self) -> None:
        pass

    def generate(self, messages, temperature=0.87):
        import ollama
        response = ollama.chat(model='llama3:8b', messages=messages, options=ollama.Options(temperature=temperature))
        message = response['message']
        return message

class Character():
    def __init__(self) -> None:
        self.dir = Path(__file__).parent / "mhistory"

        self.thoughthistory = MessagesHistory(ThoughtInstruction(), self.dir/"thought_history.json")
        self.thoughthistory.load()

        self.chathistory = MessagesHistory(ChatInstruction(), self.dir/"chat_history.json")
        self.chathistory.load()

        self.llm = LLM()
        self.user_online = True

        self.msgs_to_leave = 8
        self.max_messages_to_minimize = 20

    def minimize_context(self):
        if len(self.chathistory.messages) < self.max_messages_to_minimize:
            return
        
        dialog = ""
        for msg in self.chathistory._messages[:self.msgs_to_leave]:
            dialog += msg["content"]
            dialog += "\n"

        summary_history = MessagesHistory(SummaryInstruction())
        summary_history.append({
            'role': "user",
            'content': dialog,
        })
        summary_message = self.llm.generate(summary_history.messages, temperature=1)
        summary = summary_message["content"]

        summary_message = {
            'role': "user",
            'content': "{{context}}\n"+summary+"\n{{context}}",
        }

        self.chathistory._messages = [summary_message] + self.chathistory._messages[self.msgs_to_leave:]
        self.thoughthistory._messages = [summary_message] + self.thoughthistory._messages[self.msgs_to_leave:]

        return summary

    def construct_status_text(self):
        current_time = datetime.now()

        # Форматируем дату и время
        time_str = current_time.strftime("%H:%M")
        date_str = current_time.strftime("%A %d.%m.%Y")

        # Формируем строку
        result_str = f"its {time_str} on clock, {date_str}."
        return result_str

    def construct_thoughts_message(self, message_text=None):
        message = ""
        message += "{{status_data}} "+ self.construct_status_text() +" {{status_data}}\n"
        if message_text:
            message += "{{user_message}} "+ message_text +" {{user_message}}\n"
        
        message = message.strip()
        return {
            'role': "user",
            'content': message,
        }
    
    def construct_chat_message(self, thoughts_text=None, message_text=None):
        message = ""
        if message_text:
            message += "{{user_message}} "+ message_text +" {{user_message}}\n"
        if thoughts_text:
            message += "{{your_thoughts}} "+ thoughts_text +" {{your_thoughts}}\n"
        
        message = message.strip()
        return {
            'role': "user",
            'content': message,
        }

    def chat(self, user_text=None):
        self.minimize_context()

        message_to_thoughts = self.construct_thoughts_message(user_text)
        self.thoughthistory.append(message_to_thoughts)
        thought_message = self.llm.generate(self.thoughthistory.messages, temperature=1.5)
        self.thoughthistory.append(thought_message)
        self.thoughthistory.save()
        
        message_to_chat = self.construct_chat_message(thought_message["content"], user_text)
        self.chathistory.append(message_to_chat)
        chat_message = self.llm.generate(self.chathistory.messages, temperature=1)
        self.chathistory.append(chat_message)
        self.chathistory.save()

        text = chat_message["content"]
        return text