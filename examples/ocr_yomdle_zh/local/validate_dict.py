#!/usr/bin/env python3

import argparse
import os
import sys
import unicodedata

parser = argparse.ArgumentParser(description="""Compares nonsilence_phones and lexicon and removes words with phones not in nonsilence_phones""")
parser.add_argument('nonsilence_phones', type=str, help='File to nonsilence_phones')
parser.add_argument('lexicon', type=str, help='File to lexicon')
parser.add_argument('out_dir', type=str, help='Path to output directory')
args = parser.parse_args()

phone_list = []
phone_list.append('SIL')
with open(args.nonsilence_phones, 'r') as f:
    for line in f:
       line = line.strip()
       phone_list.append(line)

text_file = os.path.join(args.out_dir, 'lexicon_update')
text_fh = open(text_file, 'w', encoding='utf-8') 
with open(args.lexicon, 'r') as f:
    for line in f:
        line = line.strip()
        bool_valid_word = True
        for char in line.split()[1:]:
            if char not in phone_list:
                bool_valid_word = False
        if bool_valid_word:
            text_fh.write(line + '\n')
            
