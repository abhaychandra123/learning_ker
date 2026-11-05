"""Confirm that every code block in the booklet matches the repository.

Run from the repository root:  python3 booklet/check_snippets.py
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r'\\begin\{code\*\}\{(.+?)\}\n(.*?)\\end\{code\*\}', re.S)

bad = blocks = lines = 0
for chapter in sorted((ROOT / 'booklet' / 'chapters').glob('*.tex')):
    for name, body in BLOCK.findall(chapter.read_text()):
        blocks += 1
        source = {line.strip() for line in (ROOT / 'scripts' / name).read_text().splitlines()}
        for line in body.splitlines():
            text = line.strip()
            if not text:
                continue
            lines += 1
            if text not in source:
                bad += 1
                print(f'{chapter.name}: {name}: {text}')
print(f'{blocks} blocks, {lines} lines, {bad} mismatches')
sys.exit(1 if bad else 0)
