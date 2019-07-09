import os
global utt_id
from scipy.io import wavfile
import pandas as pd
import kaldi_io
import numpy as np
from generator_util import *
# 路径转化为名字
def path2name(wav_file):
    return wav_file.split('/')[-3] + wav_file.split('/')[-1].split('.')[0]
def write_wav_scp(wav_list,output_dir):
    pd_info = {'file_name':[],'path':[]}
    for wav_file in wav_list:
        if wav_file.split('.')[-1] == 'wav':

            pd_info['file_name'].append(path2name(wav_file)) #名字信息去掉后缀，如m13
            pd_info['path'].append(wav_file) #路径
    df = pd.DataFrame(pd_info)
    df = df.sort_values('file_name',ascending=True)
    df.to_csv(os.path.join(output_dir,'wav.scp'),sep=' ',index=False,header=False)
def write_utt2spk(lbe_list,output_dir):
    pd_info = {'utt_id':[],'spk_id':[]}
    for lbe_file in lbe_list:
        lbe_name = path2name(lbe_file)
        f = open(lbe_file,"rb")
        lbe_lines = f.readlines()
        f.close()
        pd_info['utt_id'] += list(map(lambda x: "%s-%d" % (lbe_name, x[0]), enumerate(lbe_lines[1:])))
        pd_info['spk_id'] += list(map(lambda x: "%s-%d" % (lbe_name, x[0]), enumerate(lbe_lines[1:])))
    df = pd.DataFrame(pd_info)
    df = df.sort_values('utt_id',ascending=True)
    df.to_csv(os.path.join(output_dir,'utt2spk'),sep=' ',index=False,header=False)
def write_spk2gender(lbe_list,output_dir):
    pd_info = {'spk_id':[],'gender':[]}
    for lbe_file in lbe_list:
        lbe_name = path2name(lbe_file)
        if lbe_name[-3] in ['m','f']:
            gender = lbe_name[-3]
        else:
            gender = lbe_name[-4]
        f = open(lbe_file,"rb")
        lbe_lines = f.readlines()
        f.close()
        pd_info['spk_id'] += list(map(lambda x: "%s-%d" % (lbe_name, x[0]), enumerate(lbe_lines[1:])))
        pd_info['gender'] += list(map(lambda x:gender,lbe_lines[1:]))

    df = pd.DataFrame(pd_info)
    df = df.sort_values('spk_id',ascending=True)
    df = df.drop_duplicates()
    df.to_csv(os.path.join(output_dir,'spk2gender'),sep=' ',index=False,header=False)
def write_seqments(lbe_list,output_dir):
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

    for lbe_file in lbe_list:
        lbe_name = path2name(lbe_file)

        f = open(lbe_file,"rb")
        lbe_lines = f.readlines()
        f.close()
        pd_info['begt'] += list(map(lambda x:time2sec(x.split(b' ')[0].decode('UTF-8')),lbe_lines[1:]))
        pd_info['endt'] += list(map(lambda x:time2sec(x.split(b' ')[1].decode('UTF-8')),lbe_lines[1:]))
        pd_info['reco_id'] += list(map(lambda x:lbe_name,lbe_lines[1:]))
        pd_info['utt_id'] += list(map(lambda x:"%s-%d"%(lbe_name,x[0]),enumerate(lbe_lines[1:])))

    df = pd.DataFrame(pd_info)
    df = df.sort_values('utt_id',ascending=True)
    df.to_csv(os.path.join(output_dir,'segments'),sep=' ',index=False,header=False)

def main():

    base_path = '/mnt/workspace1/nas_workspace2/spmiData'
    file_list = []
    lbe_list = []
    wav_list = []
    f = open('ch_datalist/863PhraseIntelbej_train_m_83-50-117.list')
    file_list += f.readlines()
    f.close()
    f = open('ch_datalist/863PhraseIntelbej_train_f_83-50-120.list')
    file_list += f.readlines()
    f.close()
    for item in file_list:
        item = item.strip()
        item = item.split('\\')
        lbe_list.append(os.path.join(base_path,item[0],'lbe',"%s.lbe"%(item[1])))
        wav_list.append(os.path.join(base_path, item[0], 'wav', "%s.wav"%(item[1])))

    output_dir = "/mnt/workspace2/yuly/Gender_chinese_train"
    dataset_dir = "/mnt/workspace2/yuly/gender_data/train_ch_raw"
    final_dir = '/mnt/workspace2/yuly/gender_data/train_ch'
    vad_dir = "/mnt/workspace2/yuly/gender_data/train_ch_vad"
    egs_dir = "/mnt/workspace2/yuly/Gender_chinese_train/egs"
    mfcc_dir = os.path.join(output_dir, "mfcc")
    mfcclog_dir = os.path.join(output_dir, "mfcclog")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(mfcc_dir):
        os.mkdir(mfcc_dir)
    if not os.path.exists(mfcclog_dir):
        os.mkdir(mfcclog_dir)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
        os.mkdir(os.path.join(dataset_dir, 'm'))
        os.mkdir(os.path.join(dataset_dir, 'f'))
    if not os.path.exists(vad_dir):
        os.mkdir(vad_dir)
        os.mkdir(os.path.join(vad_dir, 'm'))
        os.mkdir(os.path.join(vad_dir, 'f'))
    if not os.path.exists(final_dir):
        os.mkdir(final_dir)
    write_wav_scp(wav_list,output_dir)
    write_seqments(lbe_list,output_dir)
    write_utt2spk(lbe_list,output_dir)
    write_spk2gender(lbe_list,output_dir)
    print('preprocessing successfully!')
    os.chdir('/mnt/workspace2/yuly/kaldi/egs/sre16/v2')
    #os.system('utils/utt2spk_to_spk2utt.pl %s > %s'%(os.path.join(output_dir,'utt2spk'),os.path.join(output_dir,'spk2utt')))
    #os.system('steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 4 --write-utt2num-frames true --cmd \"run.pl\" %s %s %s'%(output_dir,mfcclog_dir,mfcc_dir))
    #os.system('utils/fix_data_dir.sh %s'%(output_dir))
    print('mfcc have generated successfully!')
    #os.system('sid/compute_vad_decision.sh --nj 4 --cmd \"run.pl\" %s %s %s'%(output_dir,mfcclog_dir,mfcc_dir))
    #os.system('utils/fix_data_dir.sh %s'%(output_dir))
    print('vad have generated successfully!')
    # split ark,scp to npy
    #ark2npy(output_dir,mfcc_dir,dataset_dir,vad_dir)
    # remove vad and cluster
    postprocessing(dataset_dir, vad_dir,final_dir)
if __name__ == "__main__":
    main()

