#!/usr/bin/env python3
import json
import csv
import os

ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw', 'hausa_tax_qa.json')
OUT_DIR = os.path.join(ROOT, 'data', 'processed')
OUT_CSV = os.path.join(OUT_DIR, 'hausa_tax_qa_clean.csv')

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def to_csv(data, out_path):
    fields = ['id','intent','question','answer','keywords','source']
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in data.get('qa_pairs', []):
            # keep only well-formed, tax-related pairs
            q = (item.get('question') or '').strip()
            a = (item.get('answer') or '').strip()
            if not q or not a:
                continue
            row = {k: item.get(k, '') for k in fields}
            writer.writerow(row)

def main():
    j = load_json(RAW)
    ensure_outdir(OUT_DIR)
    to_csv(j, OUT_CSV)
    print('Wrote CSV to', OUT_CSV)

if __name__ == '__main__':
    main()
