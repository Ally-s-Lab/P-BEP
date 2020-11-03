from Bio import Entrez, SeqIO
import pickle
import sys
import yaml

def main():
    with open('../config.yaml','r') as fp:
        config = yaml.load(fp,yaml.FullLoader)
    sequences = []
    Entrez.email = config['email']
    for i in range(1,25):
        i = str(i)
        sys.stdout.write('\rgetting chromosome ' + i + '/24...')
        sys.stdout.flush()
        if len(i) == 1:
            i = '0' + i
        with Entrez.efetch(db="nucleotide",
                               id='NC_0000' + i,
                               rettype="fasta") as handle:
            record = SeqIO.read(handle, "fasta")
        sequence = record.seq.__str__()
        sequences.append(sequence)
    with open('../bin/sequences.pkl', 'wb+') as fp:
        pickle.dump(sequences, fp)


if __name__ == '__main__':
    main()