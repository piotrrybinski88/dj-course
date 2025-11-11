# Tokenizer CLI Application

A command-line interface for training and using BPE (Byte Pair Encoding) tokenizers. Built with Click and the Hugging Face `tokenizers` library.

## Features

- **Train tokenizers** on custom corpora with configurable parameters
- **Encode text** using trained tokenizers
- **Test tokenizers** with sample texts
- Full command-line parameter support
- Verbose output for debugging
- Cross-platform compatibility

## Installation

### Option 1: Install as a package

```bash
pip install -e .
```

This will install the CLI as `tokenizer-cli` command globally.

### Option 2: Run directly with Python

```bash
python cli.py [command] [options]
```

## Usage

### 1. Training a Tokenizer

Train a BPE tokenizer on a corpus:

```bash
python cli.py train --corpus NKJP --output tokenizers/my_tokenizer.json --vocab-size 32000
```

**Options:**

- `--corpus TEXT` - Corpus name to use for training (default: NKJP)
- `-o, --output PATH` - Output path for the trained tokenizer (default: tokenizers/nkjp_bpe_tokenizer.json)
- `--vocab-size INTEGER` - Vocabulary size (default: 32000)
- `--min-frequency INTEGER` - Minimum token frequency (default: 2)
- `--pattern TEXT` - File pattern for corpus files (default: *.txt)
- `-v, --verbose` - Enable verbose output

**Examples:**

```bash
# Train with default settings
python cli.py train

# Train with custom vocab size and verbose output
python cli.py train --vocab-size 16000 --verbose

# Train on a specific corpus with custom output path


# Train with custom file pattern
python cli.py train --pattern "*.txt" --vocab-size 50000
```

### 2. Encoding Text

Encode text using a trained tokenizer:

```bash
python cli.py encode --tokenizer tokenizers/my_tokenizer.json "Your text here"
```

**Options:**

- `-t, --tokenizer PATH` - Path to the tokenizer JSON file (required)
- `--show-ids` - Show token IDs in output
- TEXT - Text to encode (optional, will read from stdin if not provided)

**Examples:**

```bash
# Encode text provided as argument
python cli.py encode -t tokenizers/nkjp_bpe_tokenizer.json "Litwo! Ojczyzno moja!"

# Encode text with token IDs shown
python cli.py encode -t tokenizers/nkjp_bpe_tokenizer.json "Jakże mi wesoło!" --show-ids

# Read text from stdin (interactive)
python cli.py encode -t tokenizers/nkjp_bpe_tokenizer.json
# Enter text at the prompt
```

### 3. Testing a Tokenizer

Test a tokenizer with sample Polish texts:

```bash
python cli.py test --tokenizer tokenizers/my_tokenizer.json
```

**Options:**

- `-t, --tokenizer PATH` - Path to the tokenizer JSON file (required)
- `--show-ids` - Show token IDs in output

**Examples:**

```bash
# Test tokenizer with default samples
python cli.py test -t tokenizers/nkjp_bpe_tokenizer.json

# Test with token IDs displayed
python cli.py test -t tokenizers/nkjp_bpe_tokenizer.json --show-ids
```

## Output Examples

### Training Output

```
Starting tokenizer training...
  Corpus: NKJP
  Output: tokenizers/nkjp_bpe_tokenizer.json
  Vocab size: 32000
  Min frequency: 2
Initializing BPE tokenizer...
Collecting corpus files with pattern '*.txt'...
Found 100 files
  - /path/to/file1.txt
  - /path/to/file2.txt
  - /path/to/file3.txt
  ... and 97 more
Training tokenizer (this may take a while)...
[████████████████████████████████] 100%
✓ Tokenizer trained and saved to tokenizers/nkjp_bpe_tokenizer.json
```

### Encoding Output

```
============================================================
Original text:
Litwo! Ojczyzno moja! ty jesteś jak zdrowie.

Tokens:
['Litwo', '!', 'Ojczyzno', 'moja', '!', 'ty', 'jesteś', 'jak', 'zdrowie', '.']

Token IDs:
[123, 456, 789, 234, 456, 567, 890, 234, 567, 234]
============================================================
```

### Testing Output

```
Testing tokenizer: tokenizers/nkjp_bpe_tokenizer.json

Sample 1:
  Text: Litwo! Ojczyzno moja! ty jesteś jak zdrowie.
  Tokens: ['Litwo', '!', 'Ojczyzno', 'moja', '!', 'ty', 'jesteś', 'jak', 'zdrowie', '.']
  Token count: 10

Sample 2:
  Text: Jakże mi wesoło!
  Tokens: ['Jak', 'że', 'mi', 'wesoło', '!']
  Token count: 5

Sample 3:
  Text: Jeśli wolisz mieć pełną kontrolę...
  Tokens: [...]
  Token count: 45
```

## Getting Help

View all available commands:

```bash
python cli.py --help
```

View help for a specific command:

```bash
python cli.py train --help
python cli.py encode --help
python cli.py test --help
```

## Architecture

The CLI app is structured around three main commands:

1. **train** - Initializes a BPE tokenizer, configures training parameters, loads corpus files, trains the model, and saves it
2. **encode** - Loads a trained tokenizer from file and encodes text input
3. **test** - Loads a tokenizer and tests it against predefined Polish sample texts

Each command includes:
- Input validation
- Error handling with informative messages
- Verbose logging option
- Click decorators for clean CLI interface

## Dependencies

- `tokenizers>=0.13.0` - Hugging Face tokenizers library
- `click>=8.0.0` - Command-line interface creation kit

## File Structure

```
.
├── cli.py                    # Main CLI application
├── tokenizer-build.py        # Original tokenizer training script
├── setup.py                  # Package installation configuration
├── requirements.txt          # Python dependencies
├── CLI_README.md            # This file
└── tokenizers/              # Directory for saved tokenizers
    ├── nkjp_bpe_tokenizer.json
    ├── custom_bpe_tokenizer.json
    └── ...
```

## Troubleshooting

### Command not found after installation

If you installed with `pip install -e .` but the command isn't found, try:

```bash
pip install -e . --force-reinstall
```

### Tokenizer file not found

Make sure the tokenizer path is correct and the file exists:

```bash
ls -la tokenizers/
```

### Corpus not found during training

Ensure the corpus name is correct and the corpus files are available through the `get_corpus_file()` function from the `corpora` module.

## Version

- **CLI Version:** 1.0.0
- **Python:** 3.8+
- **Click:** 8.0.0+
- **Tokenizers:** 0.13.0+

## License

MIT License
