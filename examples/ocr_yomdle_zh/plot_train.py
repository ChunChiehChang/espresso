#!/usr/bin/env python3

import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import pdb
import os
import re

parser = argparse.ArgumentParser(description="""Plots the training and validation loss""")
parser.add_argument('train_log', type=str, help='Training logs')
parser.add_argument('output', type=str, help='Output plot')
args = parser.parse_args()

train_loss = []
train_epoch = []
valid_loss = []
valid_epoch = []
with open(args.train_log, 'r') as f:
    for line in f:
        line = line.strip()
        if 'INFO' in line:
            if 'train' in line:
                pattern = re.compile(r'epoch[\s]+(?P<epoch>\d*).* loss[\s]+(?P<loss>[\d.]*)')
                m = pattern.search(line)
                if m:
                    train_epoch.append(float(m.group('epoch')))
                    train_loss.append(float(m.group('loss')))
            elif 'valid' in line:
                pattern = re.compile(r'epoch[\s]+(?P<epoch>\d*).* loss[\s]+(?P<loss>[\d.]*)')
                m = pattern.search(line)
                if m:
                    valid_epoch.append(float(m.group('epoch')))
                    valid_loss.append(float(m.group('loss')))

plt.plot(train_epoch, train_loss)
plt.plot(valid_epoch, valid_loss)
plt.savefig(args.output)
