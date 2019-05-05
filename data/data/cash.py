import os
import numpy as np
files = os.listdir('mo')
for file in files:
	file_path = os.path.join('mo',file)
	npvectors = np.load(file_path)
	length = len(npvectors)
	i = 0
	while i < length-200:
		Nfile_path = os.path.join('m',file.split('.')[0] + '{}.npy'.format(i))
		np.save(Nfile_path, npvectors[i:i+200])
		i = i + 200