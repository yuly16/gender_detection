import os
global utt_id
from scipy.io import wavfile
import pandas as pd
import kaldi_io
import numpy as np
from generator_util import *
# def time2frame(tim,fp):
# 	unit = 1/fp
# 	frame = 0
# 	if len(tim.split(':')) == 1:
# 		frame = tim.split(':')[0]
# 	elif len(tim.split(':')) == 2:

# 	else:
# 		raise ValueError('The format of the time is wrong.')
def write_wav_scp(wav_dir,output_dir):
	pd_info = {'file_name':[],'path':[]}
	for wav_file in os.listdir(wav_dir):
		if wav_file.split('.')[-1] == 'wav':
			pd_info['file_name'].append(wav_file.split('.')[0])
			pd_info['path'].append(os.path.join(wav_dir,wav_file))
	df = pd.DataFrame(pd_info)
	df = df.sort_values('file_name',ascending=True)
	df.to_csv(os.path.join(output_dir,'wav.scp'),sep=' ',index=False,header=False)
def write_utt2spk(lbe_dir,output_dir):
	pd_info = {'utt_id':[],'spk_id':[]}
	for lbe_file in os.listdir(lbe_dir):
		lbe_name = lbe_file.split('.')[0]
		lbe_path = os.path.join(lbe_dir,lbe_file)
		f = open(lbe_path,"rb")
		lbe_lines = f.readlines()
		f.close()
		pd_info['spk_id'] += list(map(lambda x:"%s-%s"%(lbe_name,x.split(b' ')[3].decode('UTF-8').split('=')[1]),lbe_lines[1:]))
		pd_info['utt_id'] += list(map(lambda x:"%s-%s-%d"%(lbe_name,x[1].split(b' ')[3].decode('UTF-8').split('=')[1],x[0]),enumerate(lbe_lines[1:])))

	df = pd.DataFrame(pd_info)
	df = df.sort_values('utt_id',ascending=True)
	df.to_csv(os.path.join(output_dir,'utt2spk'),sep=' ',index=False,header=False)
def write_spk2gender(lbe_dir,output_dir):
	id2gender = {'0':'m','1':'f'}
	pd_info = {'spk_id':[],'gender':[]}
	for lbe_file in os.listdir(lbe_dir):
		lbe_name = lbe_file.split('.')[0]
		lbe_path = os.path.join(lbe_dir,lbe_file)
		f = open(lbe_path,"rb")
		lbe_lines = f.readlines()
		f.close()
		pd_info['spk_id'] += list(map(lambda x:"%s-%s"%(lbe_name,x.split(b' ')[3].decode('UTF-8').split('=')[1]),lbe_lines[1:]))
		pd_info['gender'] += list(map(lambda x:id2gender[x.split(b' ')[2].decode('UTF-8').split('=')[1]],lbe_lines[1:]))

	df = pd.DataFrame(pd_info)
	df = df.sort_values('spk_id',ascending=True)
	df = df.drop_duplicates()
	df.to_csv(os.path.join(output_dir,'spk2gender'),sep=' ',index=False,header=False)
def write_seqments(lbe_dir,output_dir):
	pd_info = {'utt_id':[],'reco_id':[],'begt':[],'endt':[]}
	def time2sec(tim):
		tim = tim.split('=')[-1]
		if len(tim.split(':')) == 2:	
			sec = float(tim.split(':')[-1])
			minute = int(tim.split(':')[-2])
			sec = sec + 60 * minute
		elif len(tim.split(':')) == 3:
			sec = float(tim.split(':')[-1])
			minute = int(tim.split(':')[-2])
			hour = int(tim.split(':')[-3])
			sec = sec + 60 * minute + 3600 * hour
		else:
			raise ValueError('The time format is wrong!')
		sec = round(sec,2)
		return sec

	for lbe_file in os.listdir(lbe_dir):
		lbe_name = lbe_file.split('.')[0]
		lbe_path = os.path.join(lbe_dir,lbe_file)

		f = open(lbe_path,"rb")
		lbe_lines = f.readlines()
		f.close()
		pd_info['begt'] += list(map(lambda x:time2sec(x.split(b' ')[0].decode('UTF-8')),lbe_lines[1:]))
		pd_info['endt'] += list(map(lambda x:time2sec(x.split(b' ')[1].decode('UTF-8')),lbe_lines[1:]))
		pd_info['reco_id'] += list(map(lambda x:lbe_name,lbe_lines[1:]))
		pd_info['utt_id'] += list(map(lambda x:"%s-%s-%d"%(lbe_name,x[1].split(b' ')[3].decode('UTF-8').split('=')[1],x[0]),enumerate(lbe_lines[1:])))
		# for lbe_line in lbe_lines[1:]:

		# 	begt = lbe_line.split(b' ')[0].decode('UTF-8')
		# 	begt = time2sec(begt)
		# 	endt = lbe_line.split(b' ')[1].decode('UTF-8')
		# 	endt = time2sec(endt)
		# 	gender = lbe_line.split(b' ')[2].decode('UTF-8')
		# 	spk_id = lbe_line.split(b' ')[2].decode('UTF-8')
		# 	file_name = "$s_$s"%(spk_id,str(utt_id))
		# 	utt_id += 1
	df = pd.DataFrame(pd_info)
	df = df.sort_values('utt_id',ascending=True)
	df.to_csv(os.path.join(output_dir,'segments'),sep=' ',index=False,header=False)

def main():
	lbe_dir = "/mnt/workspace1/nas_workspace2/spmiData/Hub4m97/lbe"
	wav_dir = "/mnt/workspace1/nas_workspace2/spmiData/Hub4m97/wav"
	output_dir = "/mnt/workspace2/yuly/Gender_chinese_test"
	dataset_dir = "/mnt/workspace2/yuly/gender_data/test_raw"
	vad_dir = "/mnt/workspace2/yuly/gender_data/test_vad"
	final_dir = "/mnt/workspace2/yuly/gender_data/test"
	mfcc_dir = os.path.join(output_dir,"mfcc")
	mfcclog_dir = os.path.join(output_dir,"mfcclog")
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	if not os.path.exists(mfcc_dir):
		os.mkdir(mfcc_dir)
	if not os.path.exists(mfcclog_dir):
		os.mkdir(mfcclog_dir)
	if not os.path.exists(dataset_dir):
		os.mkdir(dataset_dir)
		os.mkdir(os.path.join(dataset_dir,'m'))
		os.mkdir(os.path.join(dataset_dir,'f'))
	if not os.path.exists(vad_dir):
		os.mkdir(vad_dir)
		os.mkdir(os.path.join(vad_dir,'m'))
		os.mkdir(os.path.join(vad_dir,'f'))
	if not os.path.exists(final_dir):
		os.mkdir(final_dir)
		os.mkdir(os.path.join(final_dir,'m'))
		os.mkdir(os.path.join(final_dir,'f'))
	write_wav_scp(wav_dir,output_dir)
	write_seqments(lbe_dir,output_dir)
	write_utt2spk(lbe_dir,output_dir)
	write_spk2gender(lbe_dir,output_dir)
	os.chdir('/mnt/workspace2/yuly/kaldi/egs/sre16/v2')
	# os.system('utils/utt2spk_to_spk2utt.pl %s > %s'%(os.path.join(output_dir,'utt2spk'),os.path.join(output_dir,'spk2utt')))
	# os.system('steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 4 --cmd \"run.pl\" %s %s %s'%(output_dir,mfcclog_dir,mfcc_dir))
	# os.system('utils/fix_data_dir.sh %s'%(output_dir))
	# print('mfcc have generated successfully!')
	# os.system('sid/compute_vad_decision.sh --nj 4 --cmd \"run.pl\" %s %s %s'%(output_dir,mfcclog_dir,mfcc_dir))
	# os.system('utils/fix_data_dir.sh %s'%(output_dir))
	# print('vad have generated successfully!')

	# split ark,scp to npy
	# ark2npy(output_dir,mfcc_dir,dataset_dir,vad_dir,utt_eq_spk=False)
	# remove vad
	remove_vad(dataset_dir, vad_dir,final_dir)
if __name__ == "__main__":
	main()

