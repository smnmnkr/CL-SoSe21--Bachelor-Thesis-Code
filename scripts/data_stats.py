from rich import print
from io import open
from conllu import parse_incr

from collections import Counter

data_file = open("./data/en_partut-ud-test.conllu", "r", encoding="utf-8")


POS: list = []

for tok_list in parse_incr(data_file):
    for tok in tok_list:

        POS.append(tok["upostag"])

print(len(POS))
print(dict(Counter(POS)))
