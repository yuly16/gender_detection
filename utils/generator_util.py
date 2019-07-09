import os
import pandas as pd
import kaldi_io
import numpy as np
import math
# 由mfcc,vad的ark转换为mfcc,vad的npy
def ark2npy(output_dir, mfcc_dir, dataset_dir, vad_dir, utt_eq_spk=True):
	# utt_eq_spk这个参数用于表示utt是否和spk相同。在中文训练集中，utt=spk;在中文测试集中，utt不等于spk
	spk2gender_file = os.path.join(output_dir, 'spk2gender')
	spk2gender = pd.read_csv(spk2gender_file, names=['spk_id', 'gender'], sep=' ')

	mfcc_list = os.listdir(mfcc_dir)
	for mfcc_file in mfcc_list:
		if mfcc_file.split('_')[0] == 'raw' and mfcc_file.split('.')[-1] == 'ark':
			mfcc_path = os.path.join(mfcc_dir, mfcc_file)
			for key, mat in kaldi_io.read_mat_ark(mfcc_path):
				spk = '-'.join(key.split('-')[:-1])

				utt = key
				if utt_eq_spk:
					gender = spk2gender[spk2gender['spk_id'].isin([utt])]['gender']
				else:
					gender = spk2gender[spk2gender['spk_id'].isin([spk])]['gender']
				np_path = os.path.join(dataset_dir, gender.values[0])
				# print(key,mat.shape)
				np_file = os.path.join(np_path, "{}.npy".format(utt))
				np.save(np_file, mat)
		if mfcc_file.split('_')[0] == 'vad' and mfcc_file.split('.')[-1] == 'ark':
			mfcc_path = os.path.join(mfcc_dir, mfcc_file)
			for key, mat in kaldi_io.read_vec_flt_ark(mfcc_path):
				spk = '-'.join(key.split('-')[:-1])

				utt = key
				if utt_eq_spk:
					gender = spk2gender[spk2gender['spk_id'].isin([utt])]['gender']
				else:
					gender = spk2gender[spk2gender['spk_id'].isin([spk])]['gender']
				np_path = os.path.join(vad_dir, gender.values[0])
				# print(key,mat.shape)
				np_file = os.path.join(np_path, "{}.npy".format(utt))
				np.save(np_file, mat)
def postprocessing(dataset_dir, vad_dir,output_dir):
	# the function is aimed at:
	# 1. remove the silence of mfcc based on VAD
	# 2. cluster mfcc in similar lengths and clip them into the same length

	#open utt2frame_num
	for gender in ['m','f']:
		dataset_path = os.path.join(dataset_dir,gender)
		vad_path = os.path.join(vad_dir,gender)
		for item in os.listdir(dataset_path):
			dataset_file = os.path.join(dataset_path,item)
			vad_file = os.path.join(vad_path,item)
			mfcc = np.load(dataset_file)
			vad = np.load(vad_file)
			# remove vad
			mfcc = mfcc[vad == 1]
			#cluster
			length_unit = 50
			length_category = math.floor(mfcc.shape[0] / length_unit)
			if length_category > 0:
				mfcc = mfcc[:length_category*length_unit]
				output_path = os.path.join(output_dir, str(length_category), gender)
				if not os.path.exists(os.path.join(output_dir, str(length_category))):
					os.mkdir(os.path.join(output_dir, str(length_category)))
				if not os.path.exists(output_path):
					os.mkdir(output_path)
				output_file = os.path.join(output_path,item)
				np.save(output_file,mfcc)

def ark2egs(data_dir,egs_dir):
	cmd = "sid/nnet3/xvector/get_egs.sh --cmd \"run.pl\" --nj 8 --stage 0 --frames-per-iter 1000000000 --frames-per-iter-diagnostic 500000 --min-frames-per-chunk 10 --max-frames-per-chunk 200 --num-diagnostic-archives 3 --num-repeats 40 %s %s"%(data_dir,egs_dir)
	os.system(cmd)

  
