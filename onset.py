import scipy.io as spio
import scipy.io.wavfile as spwav
from collections import namedtuple
import os
import tensorflow as tf
import random
import math
import numpy as np

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
		label_path = 'Leveau\\goodlabels\\' 
		files = os.listdir(label_path)
		#print(files)
		onsets = []

		for file in files:
			mat = spio.loadmat(label_path+file, squeeze_me=True)
			onsets.append(mat['labels_time'])

		names = [name.rstrip('.mat') for name in files]
		return dict(zip(names, onsets))

	def getAudio(self):
		audio_path = 'Leveau\\sounds\\' 
		files = os.listdir(audio_path)
		#print(files)
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

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv1d(x, W):
	return tf.nn.conv1d(x, W, 1, padding='SAME')

def max_pool(x):
	return tf.layers.max_pooling1d(x, 2, 2, padding='SAME')

def net(x):
	with tf.name_scope('reshape'):
		x_window = tf.reshape(x, [-1, WIN_SIZE, 1]) #[t sample, t length, channels]

	CONV1_SIZE = 100
	NUM_FILTERS1 = 16
	with tf.name_scope('conv1'):
		w_conv1 = weight_variable([CONV1_SIZE, 1, NUM_FILTERS1]) 
		b_conv1 = bias_variable([NUM_FILTERS1])
		o_conv1 = tf.nn.relu(conv1d(x_window, w_conv1) + b_conv1)
		#o_conv1->[batch, win_size, filter_num]

	WIN_POOL1_SIZE = math.ceil(WIN_SIZE/2)
	with tf.name_scope('pool1'):
		o_pool1 = max_pool(o_conv1)
#		o_pool1_flat = tf.reshape(o_pool1, [-1, NUM_FILTERS1*WIN_POOL1_SIZE, 1])

	CONV2_SIZE = 50
	NUM_FILTERS2 = NUM_FILTERS1*2
	with tf.name_scope('conv2'):
		w_conv2 = weight_variable([5, NUM_FILTERS1, NUM_FILTERS2])
		b_conv2 = bias_variable([NUM_FILTERS2])
		o_conv2 = tf.nn.relu(conv1d(o_pool1, w_conv2) + b_conv2)

	WIN_POOL2_SIZE = math.ceil(WIN_POOL1_SIZE/2)
	with tf.name_scope('pool2'):
		o_pool2 = max_pool(o_conv2)
		#remove conv deliniation and just make string of vectors	
		o_pool_flat = tf.reshape(o_pool2, [-1, WIN_POOL2_SIZE*NUM_FILTERS2])

	FC_SIZE = 64
	with tf.name_scope('fc1'):
		w_fc1 = weight_variable([WIN_POOL2_SIZE*NUM_FILTERS2, FC_SIZE])
		b_fc1 = weight_variable([FC_SIZE])
		o_fc1 = tf.nn.relu(tf.matmul(o_pool_flat, w_fc1) + b_fc1)

	#output layer
	with tf.name_scope('fc2'):
		w_fc2 = weight_variable([FC_SIZE, 2])
		b_fc2 = weight_variable([2])
		logits = tf.matmul(o_fc1, w_fc2) + b_fc2

	return logits

def output_layer(y, logits):
	with tf.name_scope('loss'):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)

	cross_entropy = tf.reduce_mean(cross_entropy)

	with tf.name_scope('optimizer'):
		train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

	with tf.name_scope('accuracy'):
		correctness = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)) 
		correctness = tf.cast(correctness, tf.float32)
	accuracy = tf.reduce_mean(correctness)

	return (train_step, accuracy)
	
def main():
	#671 good labels
	data = Leveau(true_multiplier=8)
	data.printStats()

	x = tf.placeholder(tf.float32, [None, WIN_SIZE]) #None specifies arbitrary number of input windows
	y = tf.placeholder(tf.float32, [None, 2])

	logits = net(x)
	train_step, accuracy = output_layer(y, logits)

	# with tf.Session().as_default() as sess:
	# 	sess.run(tf.global_variables_initializer())
	# 	batch_x, batch_y = data.getBatch('train', 50)
	# 	result = accuracy.eval(feed_dict={x: batch_x, y: batch_y})
	# 	print(result.shape)
	# exit()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		step_num = 20000
		for i in range(step_num):
			batch_x, batch_y = data.getBatch('train', 50)
			
			if i % 100 == 0:
				#evaluate on eval set
				eval_x, eval_y = data.getSet('eval')
				train_accuracy = accuracy.eval(feed_dict={x: eval_x, y: eval_y})
				print('step ', i, 'training accuracy ', train_accuracy)

			train_step.run(feed_dict={x: batch_x, y: batch_y})

		#evaluate on test set
		test_x, test_y = data.getSet('test')	
		test_accuracy = accuracy.eval(feed_dict={x: test_x, y: test_y})
		print('test accuracy ', test_accuracy) 
main()