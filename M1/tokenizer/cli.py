#!/usr/bin/env python3
"""
CLI application for tokenizer training and encoding using Click.
Supports BPE tokenizer training on various corpora.
"""

import click
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from corpora import get_corpus_file


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Tokenizer CLI - Train and use BPE tokenizers."""
    pass


@cli.command()
@click.option(
    "--corpus",
    type=str,
    default="NKJP",
    help="Corpus name to use for training (default: NKJP)"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="tokenizers/nkjp_bpe_tokenizer.json",
    help="Output path for the trained tokenizer (default: tokenizers/nkjp_bpe_tokenizer.json)"
)
@click.option(
    "--vocab-size",
    type=int,
    default=32000,
    help="Vocabulary size (default: 32000)"
)
@click.option(
    "--min-frequency",
    type=int,
    default=2,
    help="Minimum token frequency (default: 2)"
)
@click.option(
    "--pattern",
    type=str,
    default="*.txt",
    help="File pattern for corpus files (default: *.txt)"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output"
)
def train(corpus: str, output: str, vocab_size: int, min_frequency: int, pattern: str, verbose: bool):
    """Train a BPE tokenizer on the specified corpus."""
    try:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if verbose:
            click.echo(f"Starting tokenizer training...")
            click.echo(f"  Corpus: {corpus}")
            click.echo(f"  Output: {output}")
            click.echo(f"  Vocab size: {vocab_size}")
            click.echo(f"  Min frequency: {min_frequency}")

        # Initialize the Tokenizer (BPE model)
        if verbose:
            click.echo("Initializing BPE tokenizer...")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        # Set the pre-tokenizer
        tokenizer.pre_tokenizer = Whitespace()

        # Set the Trainer
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            vocab_size=vocab_size,
            min_frequency=min_frequency
        )

        # Get corpus files
        if verbose:
            click.echo(f"Collecting corpus files with pattern '{pattern}'...")
        files = [str(f) for f in get_corpus_file(corpus, pattern)]

        if not files:
            click.echo(f"Error: No files found for corpus '{corpus}' with pattern '{pattern}'", err=True)
            raise click.Abort()

        if verbose:
            click.echo(f"Found {len(files)} files")
            for f in files[:5]:
                click.echo(f"  - {f}")
            if len(files) > 5:
                click.echo(f"  ... and {len(files) - 5} more")

        # Train the Tokenizer
        click.echo("Training tokenizer (this may take a while)...")
        with click.progressbar(length=1, label="Training") as bar:
            tokenizer.train(files, trainer=trainer)
            bar.update(1)

        # Save the tokenizer
        if verbose:
            click.echo(f"Saving tokenizer to {output}...")
        tokenizer.save(output)

        click.secho(f"✓ Tokenizer trained and saved to {output}", fg="green")

    except Exception as e:
        click.secho(f"Error: {str(e)}", err=True, fg="red")
        raise click.Abort()


@cli.command()
@click.argument("text", required=False)
@click.option(
    "--tokenizer",
    "-t",
    type=click.Path(exists=True),
    required=True,
    help="Path to the tokenizer JSON file"
)
@click.option(
    "--show-ids",
    is_flag=True,
    help="Show token IDs"
)
def encode(text: Optional[str], tokenizer: str, show_ids: bool):
    """Encode text using a trained tokenizer."""
    try:
        # Load the tokenizer
        tok = Tokenizer.from_file(tokenizer)

        # If text is not provided via argument, read from stdin
        if not text:
            click.echo("Enter text to encode (press Ctrl+D when done):")
            text = click.get_text_stream('stdin').read().strip()

        if not text:
            click.echo("Error: No text provided", err=True)
            raise click.Abort()

        # Encode the text
        encoded = tok.encode(text)

        # Display results
        click.echo("\n" + "="*60)
        click.echo("Original text:")
        click.echo(text)
        click.echo("\nTokens:")
        click.echo(encoded.tokens)

        if show_ids:
            click.echo("\nToken IDs:")
            click.echo(encoded.ids)

        click.echo("="*60)

    except FileNotFoundError:
        click.echo(f"Error: Tokenizer file not found: {tokenizer}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option(
    "--tokenizer",
    "-t",
    type=click.Path(exists=True),
    required=True,
    help="Path to the tokenizer JSON file"
)
@click.option(
    "--show-ids",
    is_flag=True,
    help="Show token IDs in output"
)
def test(tokenizer: str, show_ids: bool):
    """Test a tokenizer with sample Polish texts."""
    sample_texts = [
        "Litwo! Ojczyzno moja! ty jesteś jak zdrowie.",
        "Jakże mi wesoło!",
        "Jeśli wolisz mieć pełną kontrolę nad tym, które listy są łączone (a to jest bezpieczniejsze, gdy słownik może zawierać inne klucze), po prostu prześlij listę list do spłaszczenia.",
    ]

    try:
        # Load the tokenizer
        tok = Tokenizer.from_file(tokenizer)

        click.echo(f"Testing tokenizer: {tokenizer}\n")

        for i, txt in enumerate(sample_texts, 1):
            click.echo(f"Sample {i}:")
            click.echo(f"  Text: {txt}")

            encoded = tok.encode(txt)
            click.echo(f"  Tokens: {encoded.tokens}")

            if show_ids:
                click.echo(f"  Token IDs: {encoded.ids}")

            click.echo(f"  Token count: {len(encoded.tokens)}\n")

    except FileNotFoundError:
        click.echo(f"Error: Tokenizer file not found: {tokenizer}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli()
