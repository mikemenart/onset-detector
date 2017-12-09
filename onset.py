import scipy.io as spio
import scipy.io.wavfile as spwav
from collections import namedtuple
import os
import tensorflow as tf

#0.05 seconds
#44.1k sampling rate
SAMP_RATE = 44100
WIN_SIZE = 2205

#label of form [1,0] for true or [0,1] for false
Entry = namedtuple('Entry', ['data','label'])

def getLabels():
	label_path = 'Leveau\\goodlabels\\' 
	files = os.listdir(label_path)
	#print(files)
	onsets = []

	for file in files:
		mat = spio.loadmat(label_path+file, squeeze_me=True)
		onsets.append(mat['labels_time'])

	names = [name.rstrip('.mat') for name in files]
	return dict(zip(names, onsets))

def getAudio():
	audio_path = 'Leveau\\sounds\\' 
	files = os.listdir(audio_path)
	#print(files)
	audio = []

	for file in files:
		samp_rate, frames = spwav.read(audio_path+file)
		audio.append(frames)

	names = [name.rstrip('.wav') for name in files]
	return dict(zip(names, audio))

def makeDataPairs(labels, audio):
	keys = labels.keys()
	data_pairs = []

	for key in keys:
		print(key)
		data = audio[key]
		truth = labels[key]

		truth_id = 0
		for pos in range(0, len(data), WIN_SIZE):
			time = pos/SAMP_RATE
			if(truth_id < len(truth) and (time < truth[truth_id] < time+0.05)):
				entry = Entry(data[pos:pos+WIN_SIZE], [1,0])
				truth_id += 1
				while(truth_id < len(truth) and truth[truth_id] < pos+WIN_SIZE): truth_id += 1
			else:
				entry = Entry(data[pos:pos+WIN_SIZE], [0,1])

			data_pairs.append(entry)

		#posibility of two onsets in one window causing infinite loop
		#pos = 0
		#for time in truth:
		#	while(time < pos+WIN_SIZE):
		#		data_pairs.append(entry)
		#		pos += WIN_SIZE

		#	data_pairs.append(entry)
		#	pos += WIN_SIZE

	return data_pairs



def weight_variable(shape):
	intial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv1d(x, W):
	return tf.nn.conv1d(x, W, [1,1,1], padding='SAME')

def max_pool(x):
	return tf.layers.max_pooling1d(x, [1,2,1], [1,1,1], padding='SAME')

def net(x):
	CONV1_SIZE = 5
	NUM_FILTERS = 32

	with tf.name_scope('reshape'):
		x_window = tf.reshape(x, [-1, WIN_SIZE, 1]) #[t sample, t length, channels]

	#should produce [-1, 1000, 32]
	with tf.name_scope('conv1'):
		w_conv1 = weight_variable([CONV1_SIZE, 1, NUM_FILTERS1]) 
		b_conv1 = bias_variable([NUM_FILTERS1])
		o_conv1 = tf.nn.relu(conv1d(x, w_conv1) + b_conv1)

	with tf.name_scope('pool1'):
		o_pool1 = max_pool(o_conv1)

	#dim shoud be [-1, 500, 32]
	FC_SIZE = 64
	with tf.name_scope('fc1'):
		w_fc1 = weight_variable([WIN_SIZE/2, FC_SIZE])
		b_fc1 = weight_variable([NUM_FILTERS1])

		o_pool_flat = tf.reshape(o_pool1, [-1, WIN_SIZE/2]) #remove conv deliniation and just make string of vectors
		o_fc1 = tf.nn.relu(tf.matmul(o_pool_flat, w_fc1) + b_fc1)

	#output layer
	with tf.name_scope('fc2'):
		w_fc2 = weight_variable([FC_SIZE, 2])
		b_fc1 = weight_variable([2])
		logits = tf.matmul(o_fc1, w_fc2) + b_fc2

	return logits

#def output(logits):
#	with tf.name_scope('loss'):
	
def main():
	#671 good labels
	labels = getLabels()
	audio = getAudio()
	print(labels.keys())
	data_pairs = makeDataPairs(labels, audio)
	print(len(data_pairs))
	print(data_pairs)
	exit()
	x = tf.placeholder([None, WIN_SIZE]) #None specifies arbitrary number of input windows
	y = tf.placeholder([None, 2])

	with tf.session() as sess:
		sess.run(gloal_variale_initializer)

main()