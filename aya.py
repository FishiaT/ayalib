import requests
import sseclient
import json

endpoints = {
   "models": "/v1/models",
   "props": "/props",
   "tokenize": "/tokenize",
   "chat_completions": "/v1/chat/completions",
   "embedding": "/v1/embeddings" 
}

class AyaOption:
    ignore_eos = False
    system_prompt = ""
    temperature = 0.8
    dynatemp_range = 0.0
    dynatemp_exponent = 1.0
    seed = -1
    top_k = 40
    top_p = 0.95
    min_p = 0.05
    presence_penalty = 0.0
    frequency_penalty = 0.0
    stop_strings = []
    logit_bias = []
    n_probs = 0
    min_keep = 0
    max_tokens = -1
    typical_p = 1.0
    repeat_penalty = 1.1
    repeat_last_n = 64
    penalize_nl = True
    mirostat = 0
    mirostat_tau = 5.0
    mirostat_eta = 0.1
    xtc_probability = 0.0
    xtc_threshold = 0.1
    dry_multiplier = 0.0
    dry_base = 1.75
    dry_allowed_length = 2
    dry_penalty_last_n = -1
    dry_sequence_breakers = ['\n', ':', '"', '*']
    def __init__(self, system_prompt = "You are a helpful AI assistant."):
        self.system_prompt = system_prompt
    def get_options(self):
        return {
            "stream": True,
            "temperature": self.temperature,
            "dynatemp_range": self.dynatemp_range,
            "dynatemp_exponent": self.dynatemp_exponent,
            "seed": self.seed,
            "ignore_eos": self.ignore_eos,
            "stop": self.stop_strings,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "n_predict": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "n_probs": self.n_probs,
            "min_keep": self.min_keep,
            "typical_p": self.typical_p,
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": self.repeat_last_n,
            "penalize_nl": self.penalize_nl,
            "mirostat": self.mirostat,
            "mirostat_tau": self.mirostat_eta,
            "xtc_probability": self.xtc_probability,
            "xtc_threshold": self.xtc_threshold,
            "dry_multiplier": self.dry_multiplier,
            "dry_base": self.dry_base,
            "dry_allowed_length": self.dry_allowed_length,
            "dry_penalty_last_n": self.dry_penalty_last_n,
            "dry_sequence_breakers": self.dry_sequence_breakers
        }

class Aya:
    apis = {}
    model = None
    messages = []
    storage = {}
    max_aval_ctx = 0
    used_ctx = 0
    stop = False
    inferencing = False
    def __init__(self):
        self.apis = {
            "llm": {
                "url": "",
                "api_key": ""
            },
            "embd": {
                "url": "",
                "api_key": ""
            }
        }
    def __tokenize__(self, prompt):
        r = requests.post(f"{self.apis['llm']['url']}{endpoints["tokenize"]}", headers={"Authorization": f"Bearer {self.apis['llm']['api_key']}"}, json={"content": prompt})
        r_data = json.loads(r.text)
        return len(r_data['tokens'])
    def __completions__(self, messages: list, option: AyaOption, raw_data: bool = False):
        headers = {
            "Authorization": f"Bearer {self.apis['llm']['api_key']}",
            "Accept": "text/event-stream"
        }
        body = option.get_options()
        body['messages'] = messages
        r = requests.post(f"{self.apis['llm']['url']}{endpoints["chat_completions"]}", headers=headers, json=body, stream=True)
        c = sseclient.SSEClient(r)
        for e in c.events():
            if e.data != "[DONE]":
                data = json.loads(e.data)
                if not raw_data:
                    if data['choices'][0]['delta']:
                        yield data['choices'][0]['delta']['content']
                else:
                    yield data
                if self.stop == True:
                    break
    def __embeddings__(self, text: str, encoding_format: str = "float"):
        headers = {
            "Authorization": f"Bearer {self.apis['embd']['api_key']}"
        }
        body = {
            "input": text,
            "encoding_format": encoding_format
        }
        r = requests.post(f"{self.apis['embd']['url']}{endpoints["embedding"]}", headers=headers, json=body)
        print(json.loads(r.text))
    def connect_llm(self, url, api_key = "no-api-key"):
        r = requests.get(f"{url}{endpoints["models"]}", headers={"Authorization": f"Bearer api_key"})
        r_data = json.loads(r.text)
        if "data" in r_data.keys():
            self.apis['llm']['url'] = url
            self.apis['llm']['api_key'] = api_key
            self.apis['embd']['url'] = url
            self.apis['embd']['api_key'] = api_key
        r = requests.get(f"{url}{endpoints["props"]}")
        r_data = json.loads(r.text)
        if "default_generation_settings" in r_data.keys():
            self.max_aval_ctx = r_data['default_generation_settings']['n_ctx']
            self.model = r_data['default_generation_settings']['model']
    def change_embd_model(self, url, api_key = "no-api-key"):
        self.apis['embd']['url'] = url
        self.apis['embd']['api_key'] = api_key
    def add_message(self, role: str, content: str):
        role = role.lower()
        if role not in ["system", "assistant", "user"]:
            raise Exception("Valid roles are system, assistant and user!")
        message = {
            "role": role,
            "content": content
        }
        self.messages.append(message)
    def remove_message(self, index: int):
        self.messages.pop(index)
    def get_message(self, index: int):
        return self.messages[index]
    def clear_history(self):
        msgs = self.messages
        self.messages = []
        return msgs
    def storage_add(self, key, value):
        self.storage[key] = value
    def storage_delete(self, key):
        self.storage.pop(key)
    def storage_get(self, key):
        return self.storage[key]
    def stop_inference(self):
        if self.inferencing == True:
            self.stop = True
    def inference(self, prompt: str, option: AyaOption = AyaOption()):
        self.inferencing = True
        if len(self.messages) == 0 and not option.system_prompt == "":
            self.add_message("system", option.system_prompt)
            self.used_ctx += self.__tokenize__(option.system_prompt)
        self.add_message("user", prompt)
        self.used_ctx += self.__tokenize__(prompt)
        response = ""
        for c in self.__completions__(self.messages, option):
            response += c
            yield c
        if self.stop != True:
            self.used_ctx += self.__tokenize__(response)
            self.add_message("assistant", response)
        else:
            self.stop = False
        self.inferencing = False