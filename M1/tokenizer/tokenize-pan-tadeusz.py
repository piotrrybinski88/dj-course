from tokenizers import Tokenizer
from corpora import get_corpus_file
import pandas as pd

TOKENIZERS = {
    # "bpe": "tokenizers/bpe_tokenizer.json",
    "bielik-v1": "tokenizers/bielik-v1-tokenizer.json",
    "bielik-v2": "tokenizers/bielik-v2-tokenizer.json",
    "bielik-v3": "tokenizers/bielik-v3-tokenizer.json",
    "deep-seek-v3": "tokenizers/deep-seek-v3-tokenizer.json",
    "nkjp": "tokenizers/nkjp_bpe_tokenizer.json",
    "nkjp-50k": "tokenizers/nkjp_bpe_tokenizer_50k.json",
    "all-corpora": "tokenizers/tokenizer-all-corpora.json",
    "all-corpora-50k": "tokenizers/tokenizer-all-corpora-50k.json",
    "all-corpora-16k": "tokenizers/tokenizer-all-corpora-16k.json",
    "pan-tadeusz": "tokenizers/tokenizer-pan-tadeusz.json",
    "wolnelektury": "tokenizers/tokenizer-wolnelektury.json",
}

source_txt = ""
# with open(get_corpus_file("WOLNELEKTURY", "pan-tadeusz-ksiega-1.txt")[0], 'r', encoding='utf-8') as f:
#     source_txt = f.read()
with open("/Users/piotrrybinski/Projects/dj-course/M1/korpus-mini/the-pickwick-papers-gutenberg.txt", 'r', encoding='utf-8') as f:
    source_txt = f.read()

list_of_tokens = []

for tokenizer_name, tokenizer_path in TOKENIZERS.items():
    print("-----------")
    print(tokenizer_name)
    print(tokenizer_path)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    encoded = tokenizer.encode(source_txt)

    list_of_tokens.append((tokenizer_name, len(encoded.ids)))
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Liczba tokenów: {len(encoded.ids)}")
    df = pd.DataFrame(list_of_tokens, columns=["tokenizer_name", "num_of_tokens"])
    df.sort_values("num_of_tokens", ascending=False, inplace=True)
    df.to_csv("all-tokens-the-pickwick-papers-gutenberg.csv", index=False, header=["tokenizer_name", "num_of_tokens"])
