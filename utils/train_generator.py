import os
num = 148
for i in range(1,num+1):
    os.system("python egs_reader.py --input_dir /mnt/workspace1/liyt/kaldi/egs/sre18/v2/exp/xvector_nnet_1a/egs \
    	--output_dir /mnt/workspace2/yuly/gender_data/train/egs%d --num %d"%(i,i))
    print("finish {}".format(i))
