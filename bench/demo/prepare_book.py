#!/usr/bin/env python3
"""
Download Alice in Wonderland from Project Gutenberg, clean it, and save as alice.txt.
Pure Python, no external dependencies.

Usage:
    python3 prepare_book.py [output_path]

Default output: bench/demo/alice.txt (relative to script location)
"""

import os
import re
import sys
import urllib.request

GUTENBERG_URL = "https://www.gutenberg.org/files/11/11-0.txt"

# Markers for stripping Project Gutenberg boilerplate
START_MARKERS = [
    "*** START OF THE PROJECT GUTENBERG EBOOK",
    "*** START OF THIS PROJECT GUTENBERG EBOOK",
    "*END*THE SMALL PRINT",
]
END_MARKERS = [
    "*** END OF THE PROJECT GUTENBERG EBOOK",
    "*** END OF THIS PROJECT GUTENBERG EBOOK",
    "End of the Project Gutenberg EBook",
    "End of Project Gutenberg",
]


def download_text(url: str) -> str:
    """Download text from a URL."""
    print(f"Downloading from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "quant.cpp-demo/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    # Gutenberg serves UTF-8 with BOM sometimes
    text = raw.decode("utf-8-sig", errors="replace")
    print(f"  Downloaded {len(raw):,} bytes")
    return text


def strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header and footer."""
    lines = text.splitlines(keepends=True)

    # Find start of actual content
    start_idx = 0
    for i, line in enumerate(lines):
        for marker in START_MARKERS:
            if marker in line:
                start_idx = i + 1
                # Skip blank lines right after marker
                while start_idx < len(lines) and lines[start_idx].strip() == "":
                    start_idx += 1
                break
        if start_idx > 0:
            break

    # Find end of actual content
    end_idx = len(lines)
    for i in range(len(lines) - 1, start_idx, -1):
        for marker in END_MARKERS:
            if marker in lines[i]:
                end_idx = i
                # Trim trailing blank lines
                while end_idx > start_idx and lines[end_idx - 1].strip() == "":
                    end_idx -= 1
                break
        if end_idx < len(lines):
            break

    cleaned = "".join(lines[start_idx:end_idx])

    # Normalize whitespace: collapse multiple blank lines into one
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"


def count_stats(text: str) -> dict:
    """Compute word count, line count, and estimated token count."""
    words = text.split()
    lines = text.count("\n")
    chars = len(text)
    # GPT/LLaMA tokenizers average ~1.3 tokens per word for English prose
    est_tokens = int(len(words) * 1.3)
    return {
        "words": len(words),
        "lines": lines,
        "chars": chars,
        "est_tokens": est_tokens,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(script_dir, "alice.txt")
    output_path = sys.argv[1] if len(sys.argv) > 1 else default_output

    # Download
    raw_text = download_text(GUTENBERG_URL)

    # Clean
    print("Stripping Gutenberg header/footer ...")
    cleaned = strip_gutenberg(raw_text)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    # Stats
    stats = count_stats(cleaned)
    print(f"\nSaved: {output_path}")
    print(f"  Words:            {stats['words']:>8,}")
    print(f"  Lines:            {stats['lines']:>8,}")
    print(f"  Characters:       {stats['chars']:>8,}")
    print(f"  Est. tokens:      {stats['est_tokens']:>8,}  (~1.3 tokens/word)")
    print()
    print("Ready for book_chat.sh demo.")


if __name__ == "__main__":
    main()
