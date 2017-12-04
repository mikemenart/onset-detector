import scipy.io as spio
import os

path = 'Leveau\\goodlabels\\' 
files = os.listdir(path)
print(files)

for file in files:
	mat = spio.loadmat(path+file, squeeze_me=True)
	print(mat)