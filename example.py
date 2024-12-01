# A basic CLI inference program 

import sys
import aya
aye = aya.Aya()
aye.connect_llm("http://127.0.0.1:8002")
aye.change_embd_model("http://127.0.0.1:8003")
system_prompt="""You are a helpful assistant."""
option = aya.AyaOption(system_prompt)
print(f"Model: {aye.model}\nContext Size: {aye.max_aval_ctx}\n")
aye.__embeddings__("hello world!")
while True:
    pr = input("> ")
    for i in aye.inference(pr, option):
        sys.stdout.write(i)
        sys.stdout.flush()
    print(f"\n\nUsed context: {aye.used_ctx}/{aye.max_aval_ctx} ({round(aye.used_ctx / aye.max_aval_ctx * 100, 1)}% used)")