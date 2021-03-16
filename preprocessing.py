import os
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='snips', help="data name could be snips or atis.")
parser.add_argument('--max_n', type=int, default=10000, help="number maximum of sentences.")
args = parser.parse_args()

data_name = args.data # "atis"

file_list = os.listdir("data/sequence_labelling/" + data_name)
print (file_list)

for folder in file_list:
    print('===========================================================')
    print('=========================',folder,'===========================')
    print('===========================================================')
    path = "data/sequence_labelling/" + data_name + "/" + folder + "/"
    
    in_list = [line.rstrip() for line in open(path + "seq.in")]
    out_list = [line.rstrip() for line in open(path + "seq.out")]

    output_file = open(path + folder + '.txt', 'w')

    for i,(seq_in, seq_out) in enumerate(zip(in_list,out_list)):
        print(i)
        words = seq_in.split()
        labels = seq_out.split()
        #print(words, end='')

        for (w,l) in zip(words,labels):
          output_file.write(w+" "+l+"\n")
        output_file.write("\n")
        #if i>max_n:break
      
    output_file.close()
    shutil.copy(path + folder + ".txt", "data/sequence_labelling/")
