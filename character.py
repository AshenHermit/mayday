from groq import Groq
from messages_history import MessagesHistory, ChatInstruction, ThoughtInstruction, SummaryInstruction, MetaKeysInstruction, MetaKeysSelectorInstruction
from datetime import datetime
import re
import httpx
import os
import json
from pathlib import Path
import uuid

import chromadb
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='mhistory/log.log', level=logging.INFO)

class LLM():
    def __init__(self) -> None:
        self.groq_token = os.environ["GROQ_TOKEN"]
        self.client = Groq(api_key=self.groq_token)

    def generate(self, messages, temperature=0.87, model="llama-3.1-8b-instant"):
        completion = self.client.chat.completions.create(
            model=model,
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

MemoryDir = Path(__file__).parent / "mhistory"
class LongTermMemorizer():
    def __init__(self, filename:str="ash") -> None:
        self.llm = LLM()
        self.filepath = MemoryDir / f"{filename}_tags.json"
        self.meta_keys_instruction = MetaKeysInstruction()
        self.meta_keys_selector_instruction = MetaKeysSelectorInstruction()
        self.meta_keys = {}
        self.try_to_load()

    def try_to_load(self):
        try:
            json_text = self.filepath.read_text(encoding="utf-8")
            data = json.loads(json_text)
            for key in data:
                data[key] = set(data[key])
            self.meta_keys = data
        except:
            pass

    def save(self):
        data = { key: list(self.meta_keys[key]) for key in self.meta_keys}
        json_text = json.dumps(data, ensure_ascii=False, indent=2)
        self.filepath.write_text(json_text, encoding="utf-8")

    def generate_meta_keys(self, message):
        messages = []
        messages.append(self.meta_keys_instruction.system_message)
        messages.append({
            'role': "user",
            'content': f"Ash message:\n\n{message}",
        })
        answer = self.llm.generate(messages, temperature=1, model="llama3-70b-8192")
        new_meta_keys = answer["content"]

        pattern = r'(\w+):\s*"?([^"\n]+)"?'
        matches = re.findall(pattern, new_meta_keys)
        key_value_pairs = {key: value for key, value in matches}

        for key in key_value_pairs.keys():
            if key in self.meta_keys:
                self.meta_keys[key].add(key_value_pairs[key])
            else:
                self.meta_keys[key] = set()
                self.meta_keys[key].add(key_value_pairs[key])

        return new_meta_keys, self.meta_keys
    
    def get_meta_tags_for_messages(self, messages:str):
        messages = []
        messages.append(self.meta_keys_selector_instruction.system_message)
        messages.append({
            'role': "user",
            'content': f"all_tags:\n{", ".join(self.meta_keys.keys())}\n\nmessage:\n{messages}",
        })
        answer = self.llm.generate(messages, temperature=1, model="llama3-70b-8192")
        selected_meta_keys = answer["content"]
        selected_meta_keys = [meta_key.strip() for meta_key in selected_meta_keys.split("\n")]
        tags_text = {key: list(self.meta_keys[key])[-1] for key in selected_meta_keys}
        return tags_text

class Character():
    def __init__(self) -> None:
        self.dir = MemoryDir

        self.thoughthistory = MessagesHistory(ThoughtInstruction(), self.dir/"thought_history.json")
        self.thoughthistory.load()

        self.chathistory = MessagesHistory(ChatInstruction(), self.dir/"chat_history.json")
        self.chathistory.load()

        self.llm = LLM()
        self.user_online = True

        self.msgs_to_leave = 8
        self.max_messages_to_minimize = 20

        self.all_messages_file = self.dir/"all_messages.json"

        self.chromadb = chromadb.Client()
        self.dbcollection = self.chromadb.get_or_create_collection(name="chat_messages")
        self.load_messages_to_db()

    def load_messages_to_db(self):
        data = json.loads(self.all_messages_file.read_text(encoding="utf-8"))
        for i, message in enumerate(data):
            self.add_message_to_db(message, i)

    def add_message_to_db(self, message_obj, id=None):
        if id is None: id = uuid.uuid4().int
        self.dbcollection.add(
            documents=[f"from: \"{message_obj["text"]}\": "+message_obj["text"]],
            ids=[str(id)]
        )

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
        summary_message = self.llm.generate(summary_history.messages, temperature=1, model="llama3-70b-8192")
        summary = summary_message["content"]

        summary_message = {
            'role': "user",
            'content': "---\n{{context}}\n"+summary+"\n\n",
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
        message += "---\n{{status_data}}\n"+ self.construct_status_text() +"\n\n"
        if message_text:
            message += "---\n{{user_message}}\n"+ message_text +"\n\n"
        
        message = message.strip()
        return {
            'role': "user",
            'content': message,
        }

    def get_memories(self, user_message):
        text = ""
        query_message = user_message
        results = self.dbcollection.query(
            query_texts=[query_message],
            n_results=3
        )
        for doc in results['documents'][0][:3]:
            text += doc
            text += "\n"
        return text
    
    def construct_chat_message(self, thoughts_text=None, message_text=None):
        message = ""
        message += "---\n{{status_data}}\n"+ self.construct_status_text() +"\n\n"
        if message_text:
            message += "---\n{{user_message}}\n"+ message_text +"\n\n"
        if thoughts_text:
            message += "---\n{{your_thoughts}}\n"+ thoughts_text +"\n\n"
        
        message = message.strip()
        return {
            'role': "user",
            'content': message,
        }
    
    def add_text_to_all_messages(self, data2add):
        data = json.loads(self.all_messages_file.read_text(encoding="utf-8"))
        self.add_message_to_db(data2add)
        data.append(data2add)
        self.all_messages_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def chat(self, user_text=None):
        self.minimize_context()

        if not user_text:
            message_to_thoughts = self.construct_thoughts_message(user_text)
            self.thoughthistory.append(message_to_thoughts)
            thought_message = self.llm.generate(self.thoughthistory.messages, temperature=1.4)
            self.thoughthistory.append(thought_message)
            self.thoughthistory.save()
        
            message_to_chat = self.construct_chat_message(thought_message, user_text)
        else:
            message_to_chat = self.construct_chat_message(None, user_text)
            self.add_text_to_all_messages({"from": "Ash", "text": user_text})

        self.chathistory.append(message_to_chat)
        messages = self.chathistory.messages
        messages[-1]["content"] = f"memories:\n{self.get_memories(user_text)}\n---\n" + messages[-1]["content"]
        logger.info(messages[-1]["content"])

        chat_message = self.llm.generate(messages, temperature=1)
        text:str = chat_message["content"]
        if text.lower().startswith("I cannot continue this conversation"):
            chat_message = self.llm.generate(messages, temperature=1, model="llama3-70b-8192")
            text:str = chat_message["content"]
        self.chathistory.append(chat_message)
        self.chathistory.save()

        self.add_text_to_all_messages({"from": "May", "text":{text}})

        return text