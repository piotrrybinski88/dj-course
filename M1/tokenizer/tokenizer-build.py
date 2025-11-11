from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from corpora import get_corpus_file

# TODO change TOKENIZER_OUTPUT_FILE and NKJP to parameter
TOKENIZER_OUTPUT_FILE = "tokenizers/tokenizer-all-corpora.json"
list_of_corpus = ["NKJP", "WOLNELEKTURY", "SPICHLERZ"]

# 1. Initialize the Tokenizer (BPE model)
tokenizer = Tokenizer(BPE(unk_token="[UNK]")) 

# 2. Set the pre-tokenizer (e.g., split on spaces)
tokenizer.pre_tokenizer = Whitespace()

# 3. Set the Trainer
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=32000,
    min_frequency=2
)

FILES = [str(f) for corpus_name in list_of_corpus for f in get_corpus_file(corpus_name, "*.txt")]


print(FILES)

# 4. Train the Tokenizer
tokenizer.train(FILES, trainer=trainer)

# 5. Save the vocabulary and tokenization rules
tokenizer.save(TOKENIZER_OUTPUT_FILE)

for txt in [
    "Litwo! Ojczyzno moja! ty jesteś jak zdrowie.",
    "Jakże mi wesoło!",
    "Jeśli wolisz mieć pełną kontrolę nad tym, które listy są łączone (a to jest bezpieczniejsze, gdy słownik może zawierać inne klucze), po prostu prześlij listę list do spłaszczenia.",
]:
    encoded = tokenizer.encode(txt)
    print("Zakodowany tekst:", encoded.tokens)
    print("ID tokenów:", encoded.ids)
