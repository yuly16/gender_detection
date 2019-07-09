py文件说明
==============
>main_multi.py: 多数据集训练，中文测试集测试<br>
>main_single.py: 单数据集训练，中文数据集测试<br>
>models:
>>models.py: 模型代码<br>

>data:
>>mfcc_dataloader.py: 数据集生成代码<br>
>log: 模型保存以及记录<br>

>utils:
>> train_generator:产生英文训练集<br>
>> train_generator_ch: 产生中文训练集<br>
>> test_generator: 产生中文测试集<br>
>> egs_reader: 产生英文训练集的依赖文件<br>
>> generator_util: 采用lbe,wav方式产生的数据集所依赖的一些函数<br>
>> util.py: dataloaders集合，保存模型，保存训练准确率和loss<br>
>> ch_datalist:保存了中文训练集所需要的数据集索引。

英文训练集生成
==============
1. Preparing train data from sre, sre10, swbd and fisher:
        1) cd gender/utils
        2) run the script "train_generator.py" without any agrument
    The npy training data is generated in the directory /mnt/workspace2/yuly/gender_data/train
2. Training the model:
        run the script main.py


中文训练集生成
==============
1.从wav到scp, ark:
-----------------
**注意：一定要按照下面的排序方式生成文件，否则会报错！**<br>
需要准备如下文件：
>segments: `句子id` `wav_id` `开始时间` `终止时间` <br> 
>>注：wav_id表示wav的identity,所以可以用文件名代替<br>
>>例子：863f00-0 863f00 0.96 4.7<br>
>>排序方式为utt_id

>wav.scp: `wav_id` `wav路径`
>>排序方式为file_name

>utt2spk: `句子id` `句子id`
>>注：这组数据集中我们没有说话人的id信息，所以保持utt_id和spk_id一样就可以；以utt_id排序

>spk2gender: `句子id` `gender`
>>gender的格式为m or f，排序方式为utt_id


## 2.生成mfcc和vad
代码如下：
```python
    os.chdir('/mnt/workspace2/yuly/kaldi/egs/sre16/v2')
    os.system('utils/utt2spk_to_spk2utt.pl %s > %s'%(os.path.join(output_dir,'utt2spk'),os.path.join(output_dir,'spk2utt')))
    os.system('steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 4 --cmd \"run.pl\" %s %s %s'%(output_dir,mfcclog_dir,mfcc_dir))
    os.system('utils/fix_data_dir.sh %s'%(output_dir))
    os.system('sid/compute_vad_decision.sh --nj 4 --cmd \"run.pl\" %s %s %s'%(output_dir,mfcclog_dir,mfcc_dir))
    os.system('utils/fix_data_dir.sh %s'%(output_dir))
```
代码分析：
>1. 一定要切换到相应目录！否则下面kaldi的脚本运行不了<br>
>2. 产生spk2utt文件，这个文件是kaldi数据准备的必不可少的文件<br>
>3. 生成mfcc频谱
>4. 保证feat.scp，wav.scp，vad.scp的内容一样。对这个文件的理解还不多，以后再填坑
>5. 产生vad数据。每一个utt的vad和mfcc是一一对应的。vad是一个vector，mfcc是一个matrix。vad的shape和mfcc的shape[0]是一样的。vad∈{0,1}，mfcc的某一个元素对应vad的位置为0，则说明mfcc的这个位置是静音。

## 3.由mfcc,vad到npy
调用ark2npy(output_dir,mfcc_dir,dataset_dir,vad_dir)函数即可。

## 4.后处理
后处理包括两部分:除去静音；多label聚类。
调用postprocessing函数。
