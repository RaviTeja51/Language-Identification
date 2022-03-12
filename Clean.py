import re
import pandas as pd

sent_begin =  re.compile(r"\#\s\bsent_enum\b\s=\s(?P<sentence_number>\d+)\n")
sent_end =  re.compile(r"^\n$")

text = []
tags = []
with open('dev.conll') as f:
    line = f.readline()
    while line:
        if sent_begin.match(line):
            sent = []
            label = []
            print(f"""Reading sentnence:{sent_begin.match(line).group("sentence_number")}""")
            line = f.readline()
            while not sent_end.match(line):
                word,tag = line.strip("\n").split("\t")
                sent.append(word.lower())
                label.append(tag)
                line = f.readline()
            text.append(" ".join(sent))
            tags.append(" ".join(label))
        line = f.readline()

df = pd.DataFrame({"text":text,"labels":tags})
df.to_csv("dev.csv",index=False)


text = []
tags = []
with open('train.conll') as f:
    line = f.readline()
    while line:
        if sent_begin.match(line):
            sent = []
            label = []
            print(f"""Reading sentnence:{sent_begin.match(line).group("sentence_number")}""")
            line = f.readline()
            while not sent_end.match(line):
                word,tag = line.strip("\n").split("\t")
                sent.append(word.lower())
                label.append(tag)
                line = f.readline()
            text.append(" ".join(sent))
            tags.append(" ".join(label))
        line = f.readline()

df = pd.DataFrame({"text":text,"labels":tags})
df.to_csv("train.csv",index=False)