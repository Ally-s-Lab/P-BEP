import pickle
import os
import numpy as np

with open('sequences.pkl', 'rb') as fp:
        sequences = pickle.load(fp)

BUFFER = 70
WINDOW = 25
MEMO_FILE = 'positions_prob_memo.pkl'
if not os.path.isfile(MEMO_FILE):
    dic = {}
    with open(MEMO_FILE,'wb+') as fp:
        pickle.dump(dic,fp)

def coord_to_prob(chromosome,position,window=WINDOW):
    chromosome = str(chromosome).lower()
    position = int(position)
    if chromosome == 'x':
        chromosome = 23
    if chromosome == 'y':
        chromosome = 24
    try:
        chromosome = int(chromosome) - 1
    except ValueError:
        print(chromosome,position)
        return -1
    position -= 1
    try:
        site = str(sequences[chromosome][position]).upper()
    except IndexError:
        print('index error')
        return -1
    if str(site) != 'A' and str(site) != 'T':
        print('Value error')
        return -1
    try:
        sequence = sequences[chromosome][position-window-BUFFER:position+window+BUFFER+1].seq
    except IndexError:
        print('index error')
        return -1
    if site == 'T':
        sequence = str(sequence.reverse_complement()).upper()[::-1]
    else:
        sequence = str(sequence).upper()
    sequence = sequence.__str__()
    os.system("""echo "{}" | ../ViennaRNA-2.4.15/src/bin/RNAplfold -u 1 -o""".format(sequence))
    result = []
    with open('plfold_lunp','r') as fp:
        lines = fp.readlines()[2:]
    for i in range(BUFFER,BUFFER+window+1+window):
        try:
            line = lines[i].split('\t')
        except IndexError:
            return -1
        prob = float(line[1])
        result.append(prob)
    return np.array(result)


