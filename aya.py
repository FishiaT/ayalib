from utils import endpoints, chat_templates
from pathlib import Path
import requests
import sseclient
import json
import datetime

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
    url = None
    api_key = None
    chat_template = {}
    messages = []
    max_aval_ctx = 0
    used_ctx = 0
    def __init__(self, url, api_key, chat_template):
        self.chat_template = chat_templates.get_template(chat_template)
        r = requests.get(f"{url}{endpoints.models}", headers={"Bearer": api_key})
        r_data = json.loads(r.text)
        if "data" in r_data.keys():
            self.url = url
            self.api_key = api_key
            self.__dryrun__()
    def __build_prompt__(self):
        prompt = ""
        for msg in self.messages:
            prompt += f"{self.chat_template[msg['role']]}{msg['content']}{self.chat_template['suffix']}"
        return prompt
    def __tokenize__(self, prompt):
        r = requests.post(f"{self.url}{endpoints.tokenize}", headers={"Bearer": self.api_key}, json={"content": prompt})
        r_data = json.loads(r.text)
        return len(r_data['tokens'])
    def __completions__(self, prompt: str, option: AyaOption, raw_data: bool = False):
        headers = {
            "Bearer": self.api_key,
            "Accept": "text/event-stream"
        }
        body = option.get_options()
        body['prompt'] = prompt
        r = requests.post(f"{self.url}{endpoints.completion}", headers=headers, json=body, stream=True)
        c = sseclient.SSEClient(r)
        for e in c.events():
            data = json.loads(e.data)
            if not raw_data:
                yield data['content']
            else:
                yield data
    def __dryrun__(self):
        option = AyaOption("")
        option.max_tokens = 1
        r = {}
        for i in self.__completions__("DRYRUN", option, True):
            r = i
        self.max_aval_ctx = r['generation_settings']['n_ctx']
    def add_message(self, role: str, content: str):
        role = role.lower()
        if role not in ["system", "assistant", "user"]:
            raise Exception("Valid roles are system, assistant and user!")
        message = {
            "role": role,
            "content": content,
            "time": str(int(round(datetime.datetime.now().timestamp())))
        }
        self.messages.append(message)
    def remove_message(self, index: int):
        self.messages.pop(index)
    def clear_history(self):
        self.messages = []
    def inference(self, prompt: str, option: AyaOption = AyaOption()):
        if len(self.messages) == 0 and not option.system_prompt == "":
            self.add_message("system", option.system_prompt)
            self.used_ctx += self.__tokenize__(option.system_prompt)
        self.add_message("user", prompt)
        self.used_ctx += self.__tokenize__(prompt)
        prompt = self.__build_prompt__()
        prompt += f"{prompt}{self.chat_template['assistant']}"
        completion = self.__completions__(prompt, option)
        response = ""
        for c in completion:
            response += c
            yield c
        self.add_message("assistant", response)