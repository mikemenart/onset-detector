from collections import namedtuple
import os
import scipy.io.wavfile as spwav
import scipy.io as spio
import numpy as np
import random

#0.05 seconds
#44.1k sampling rate
SAMP_RATE = 44100
WIN_SIZE = 2206

#label of form [1,0] for true or [0,1] for false
Entry = namedtuple('Entry', ['data','label'])

class Leveau:
	def __init__(self, true_multiplier = 4):
		self.true_mult = true_multiplier
		self.audio = self.getAudio()
		self.labels = self.getLabels()
		self.data_pairs = self.makeDataPairs()
		random.shuffle(self.data_pairs)

		#segment into train eval test
		train_size = int(0.6*len(self.data_pairs))
		eval_cutoff = int(0.8*len(self.data_pairs))
		train_set = self.data_pairs[:train_size]
		eval_set = self.data_pairs[train_size:eval_cutoff]
		test_set = self.data_pairs[eval_cutoff:]

		self.sets = {'train': train_set, 'eval': eval_set, 'test': test_set}

	def getBatch(self, set, size):
		batch = random.sample(self.sets[set], size)
		batch_x = [entry.data for entry in batch]
		batch_y = [entry.label for entry in batch]

		return (batch_x, batch_y)

	def getSet(self, set):
		'''Set: String that is either 'train', 'eval', or 'test' '''
		data = self.sets[set]
		set_x = [entry.data for entry in data]
		set_y = [entry.label for entry in data]
		return (set_x, set_y)

	def printStats(self):
		y = [entry.label for entry in self.data_pairs]
		num_true = 0

		for entry in y:
			if entry[0] == 1:
				num_true += 1

		print("num true ", num_true)
		print("total ", len(y))
		print("Percent True ", 100*num_true/len(y))

	def getLabels(self):
		good_path = 'Leveau\\goodlabels\\' 
		PL_path = 'Leveau\\labelsPL\\' 
		files = os.listdir(good_path)
		onsets = []

		for file in files:
			mat_good = spio.loadmat(good_path+file, squeeze_me=True)
			mat_PL = spio.loadmat(PL_path+file, squeeze_me=True)
			combined = np.append(mat_good['labels_time'], mat_PL['labels_time'])
			onsets.append(combined)

		names = [name.rstrip('.mat') for name in files]
		return dict(zip(names, onsets))

	def getAudio(self):
		audio_path = 'Leveau\\sounds\\' 
		files = os.listdir(audio_path)
		audio = []

		for file in files:
			samp_rate, frames = spwav.read(audio_path+file)
			audio.append(frames)

		names = [name.rstrip('.wav') for name in files]
		return dict(zip(names, audio))

	def makeDataPairs(self):

		def getWinData(data, pos):
			pos = max(0, pos)
			if(pos+WIN_SIZE < len(data)):
				return data[pos:pos+WIN_SIZE]
			else:
				return np.append(data[pos:], np.zeros(WIN_SIZE-(len(data)-pos)))

		def onsetInWindow(labels, pos):
			start_time = pos/SAMP_RATE
			end_time = start_time + WIN_SIZE/SAMP_RATE

			for label in labels:
				if(start_time < label < end_time):
					return True

			return False

		keys = self.labels.keys()
		data_pairs = []

		for key in keys:
			data = self.audio[key]
			truth = self.labels[key]

			#create negative samples
			for pos in range(0, len(data), int(WIN_SIZE/2)):
				if not (onsetInWindow(truth, pos)):
					entry = Entry(getWinData(data,pos), [0,1])
					data_pairs.append(entry)

			#create positive samples
			for onset in truth:
				pos = int(onset*SAMP_RATE)
				for i in range(self.true_mult):
					win_start = int(pos - WIN_SIZE*(i/self.true_mult)*0.8)
					entry = Entry(getWinData(data, win_start), [1,0])

					data_pairs.append(entry)

		print("len datapairs ", len(data_pairs))
		for pair in data_pairs:
			assert(len(pair.data) == WIN_SIZE)
		return data_pairs
