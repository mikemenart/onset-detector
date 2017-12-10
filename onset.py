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

	def __init__(self):
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
		keys = self.labels.keys()
		data_pairs = []

		def inBound(time, truth, truth_id):
			win_time = WIN_SIZE/SAMP_RATE
			return (truth_id < len(truth) 
				and (time < truth[truth_id] < time+win_time)) 

		def getWinData(data, pos):
			if(pos+WIN_SIZE < len(data)):
				return data[pos:pos+WIN_SIZE]
			else:
				return np.append(data[pos:], np.zeros(WIN_SIZE-(len(data)-pos)))

		for key in keys:
			data = self.audio[key]
			truth = self.labels[key]

			truth_id = 0
			for pos in range(0, len(data), WIN_SIZE):
				time = pos/SAMP_RATE
				if inBound(time, truth, truth_id):
					entry = Entry(getWinData(data,pos), [1,0])
					truth_id += 1
					while(truth_id < len(truth) and truth[truth_id] < pos+WIN_SIZE): truth_id += 1
				else:
					entry = Entry(getWinData(data,pos), [0,1])

				data_pairs.append(entry)

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
	CONV1_SIZE = 5
	NUM_FILTERS1 = 32
	WIN_POOL_SIZE = math.ceil(WIN_SIZE/2)

	with tf.name_scope('reshape'):
		x_window = tf.reshape(x, [-1, WIN_SIZE, 1]) #[t sample, t length, channels]

	with tf.name_scope('conv1'):
		w_conv1 = weight_variable([CONV1_SIZE, 1, NUM_FILTERS1]) 
		b_conv1 = bias_variable([NUM_FILTERS1])
		o_conv1 = tf.nn.relu(conv1d(x_window, w_conv1) + b_conv1)

	with tf.name_scope('pool1'):
		o_pool1 = max_pool(o_conv1)

	FC_SIZE = 64
	with tf.name_scope('fc1'):
		w_fc1 = weight_variable([WIN_POOL_SIZE*NUM_FILTERS1, FC_SIZE])
		b_fc1 = weight_variable([FC_SIZE])
		#should be NUM_FILTERS vectors of length WIN_POOL_SIZE
		o_pool_flat = tf.reshape(o_pool1, [-1, WIN_POOL_SIZE*NUM_FILTERS1]) #remove conv deliniation and just make string of vectors
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
		train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

	with tf.name_scope('accuracy'):
		correctness = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)) 
		correctness = tf.cast(correctness, tf.float32)
	accuracy = tf.reduce_mean(correctness)

	return (train_step, accuracy)
	
def main():
	#671 good labels
	data = Leveau()


	x = tf.placeholder(tf.float32, [None, WIN_SIZE]) #None specifies arbitrary number of input windows
	y = tf.placeholder(tf.float32, [None, 2])

	logits = net(x)
	train_step, accuracy = output_layer(y, logits)

	# with tf.Session().as_default() as sess:
	# 	sess.run(tf.global_variables_initializer())
	# 	batch = random.sample(train_set, 50) #without replacement, but how?
	# 	batch_x = [entry.data for entry in batch]
	# 	batch_y = [entry.label for entry in batch]
	# 	result = accuracy.eval(feed_dict={x: batch_x, y: batch_y})
	# 	print(result)
	# exit()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		step_num = 500
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