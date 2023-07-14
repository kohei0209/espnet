from pathlib import Path

from tqdm import tqdm
from transformers import WhisperProcessor


def normalize(model, text, output):
    processor = WhisperProcessor.from_pretrained(model)
    if output:
        output = open(output, "w")
    iterable = tqdm(text) if output else text
    for uid, transcription in iterable:
        transcription = processor.tokenizer._normalize(transcription)
        if output is None:
            print(f"({uid}) {transcription}")
        else:
            output.write(f"{uid} {transcription}\n")
    if output:
        output.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "text_file", type=str, help="Path to the text file or content of the text file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="-",
        help="Path to the output file for writing transcripts. "
        "If is '-', then write to stdout.",
    )
    parser.add_argument("--model", type=str, default="openai/whisper-large-v2")
    args = parser.parse_args()

    text_file = Path(args.text_file)
    if text_file.is_file():
        text = [
            line.strip().split(maxsplit=1)
            for line in text_file.read_text().splitlines()
        ]
    else:
        text = [("STDIN", args.text_file.strip())]
    output_file = Path(args.output_file)
    if output_file == Path("-"):
        output_file = None
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    normalize(args.model, text, output_file)
