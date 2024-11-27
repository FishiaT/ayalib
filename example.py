import aya
aye = aya.Aya("http://127.0.0.1:8002", "", "llama3")
for i in aye.inference("Hello!"):
    print(i)