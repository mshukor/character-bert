### from https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.custom_classification.md

import argparse
import os
import random
from glob import glob
from tqdm import tqdm

random.seed(0)

def main(args):
    for split in ['train', 'test']:
        samples = []
        print(split)
        for class_label in ['pos', 'neg']:
            fnames = glob(os.path.join(args.datadir, split, class_label) + '/*.txt')
            for fname in fnames:
                with open(fname) as fin:
                    line = fin.readline()
                    label = "__label__positive " if class_label == 'pos' else "__label__negative "
                    new_line = label + line
                    samples.append(new_line) #(line, 1 if class_label == 'pos' else 0))
        random.shuffle(samples)
        out_fname = 'train' if split == 'train' else 'test'
        f1 = open(os.path.join(args.datadir, out_fname + '.txt'), 'w')
        # f2 = open(os.path.join(args.datadir, out_fname + '.label'), 'w')
        for sample in tqdm(samples):
            f1.write(sample + '\n')
            # f2.write(str(sample[1]) + '\n')
        f1.close()
        # f2.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='/home/shukor/Workspace/character-bert/data/classification/imdb')
    args = parser.parse_args()
    main(args)