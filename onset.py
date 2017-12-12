import tensorflow as tf
import random
import math
from Leveau import Leveau

#0.05 seconds
#44.1k sampling rate
SAMP_RATE = 44100/2
WIN_SIZE = 2206

class Network:
	def __init__(self):
		self.x = tf.placeholder(tf.float32, [None, WIN_SIZE]) #None specifies arbitrary number of input windows
		self.y = tf.placeholder(tf.float32, [None, 2])

		self.logits, self.keep_prob = self.net_body()
		self.train_step, self.accuracy, self.loss = self.output_layer()

		self.summary_writer = tf.summary.FileWriter('log/', graph=tf.get_default_graph())
		tf.summary.scalar("Loss", self.loss)
		tf.summary.scalar("accuracy", self.accuracy)
		self.merged = tf.summary.merge_all()

	def net_body(self):
		####Utility functions for network constructoin###
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

		def batch_norm(x, win_size, filter_num):
			batch_mean, batch_var = tf.nn.moments(x, [0])
			beta = tf.Variable(tf.zeros([win_size, filter_num]))
			scale = tf.Variable(tf.ones([win_size, filter_num]))
			return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, 1e-3)

		###Network Body###
		with tf.name_scope('reshape'):
			x_window = tf.reshape(self.x, [-1, WIN_SIZE, 1]) #[t sample, t length, channels]

		CONV1_SIZE = 220 #882 #441
		NUM_FILTERS1 = 8
		with tf.name_scope('conv1'):
			w_conv1 = weight_variable([CONV1_SIZE, 1, NUM_FILTERS1]) 
			b_conv1 = bias_variable([NUM_FILTERS1])
			o_conv1 = tf.nn.relu(conv1d(x_window, w_conv1) + b_conv1)
			#o_conv1->[batch, win_size, filter_num]

		WIN_POOL1_SIZE = math.ceil(WIN_SIZE/2)
		with tf.name_scope('pool1'):
			o_pool1 = max_pool(o_conv1)

		# with tf.name_scope('batch_norm1'):
		# 	o_batch1 = batch_norm(o_pool1, WIN_POOL1_SIZE, NUM_FILTERS1)

		CONV2_SIZE = 220 #882 #22 #200
		NUM_FILTERS2 = NUM_FILTERS1
		with tf.name_scope('conv2'):
			w_conv2 = weight_variable([CONV2_SIZE, NUM_FILTERS1, NUM_FILTERS2])
			b_conv2 = bias_variable([NUM_FILTERS2])
			o_conv2 = tf.nn.relu(conv1d(o_pool1, w_conv2) + b_conv2)

		WIN_POOL2_SIZE = math.ceil(WIN_POOL1_SIZE/2)
		with tf.name_scope('pool2'):
			o_pool2 = max_pool(o_conv2)

		# with tf.name_scope('batch_norm2'):
		# 	o_batch2 = batch_norm(o_pool2, WIN_POOL2_SIZE, NUM_FILTERS2)

		# CONV3_SIZE = 882 #11 #20
		# NUM_FILTERS3 = NUM_FILTERS2
		# with tf.name_scope('conv3'):
		# 	w_conv3 = weight_variable([CONV3_SIZE, NUM_FILTERS2, NUM_FILTERS3])
		# 	b_conv3 = bias_variable([NUM_FILTERS3])
		# 	o_conv3 = tf.nn.relu(conv1d(o_batch2, w_conv3) + b_conv3)

		# WIN_POOL3_SIZE = math.ceil(WIN_POOL2_SIZE/2)
		# with tf.name_scope('pool3'):
		# 	o_pool3 = max_pool(o_conv3)

		# with tf.name_scope('batch3'):
		# 	o_batch3 = batch_norm(o_pool3, WIN_POOL3_SIZE, NUM_FILTERS3)
		# CONV4_SIZE = 5
		# NUM_FILTERS4 = NUM_FILTERS3
		# with tf.name_scope('conv4'):
		# 	w_conv4 = weight_variable([CONV3_SIZE, NUM_FILTERS3, NUM_FILTERS4])
		# 	b_conv4 = bias_variable([NUM_FILTERS4])
		# 	o_conv4 = tf.nn.relu(conv1d(o_conv3, w_conv4) + b_conv3)
		FC_SIZE = 64
		with tf.name_scope('fc1'):
			#remove conv deliniation and just make string of vectors	
			o_pool_flat = tf.reshape(o_pool2, [-1, WIN_POOL2_SIZE*NUM_FILTERS2])

			w_fc1 = weight_variable([WIN_POOL2_SIZE*NUM_FILTERS2, FC_SIZE])
			b_fc1 = weight_variable([FC_SIZE])
			o_fc1 = tf.nn.relu(tf.matmul(o_pool_flat, w_fc1) + b_fc1)

		with tf.name_scope('dropout'):
			keep_prob = tf.placeholder(tf.float32)
			o_fc1_drop = tf.nn.dropout(o_fc1, keep_prob)
		
		with tf.name_scope('fc2'):
			w_fc2 = weight_variable([FC_SIZE, 2])
			b_fc2 = weight_variable([2])
			logits = tf.matmul(o_fc1_drop, w_fc2) + b_fc2

		return (logits, keep_prob)

	def output_layer(self):
		with tf.name_scope('loss'):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)

		cross_entropy = tf.reduce_mean(cross_entropy)

		with tf.name_scope('optimizer'):
			train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

		with tf.name_scope('accuracy'):
			correctness = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1)) 
			correctness = tf.cast(correctness, tf.float32)
		accuracy = tf.reduce_mean(correctness)

		return (train_step, accuracy, cross_entropy)

	def batchEval(self, data_x, data_y, keep_prob=1, batch_size=256):
		num_samples = len(data_x)
		accuracy = 0
		for i in range(0, num_samples, batch_size):
			batch_end = min(num_samples, i+batch_size)
			accuracy +=  self.accuracy.eval(feed_dict={
				self.x: data_x[i:batch_end], 
				self.y: data_y[i:batch_end], 
				self.keep_prob: keep_prob})

		num_batches = math.ceil(num_samples/batch_size)	
		accuracy /= num_batches
		return accuracy

	def trainStep(self, batch_x, batch_y, keep_prob=0.5):
		self.train_step.run(feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 0.5})


#DATALABELS ARENT NECCESARILY ALL ONSETS, INVALIDATES DATA CREATION	
def main():
	data = Leveau(true_multiplier=2)
	data.printStats()

	net = Network()	
	# with tf.Session().as_default() as sess:
	# 	sess.run(tf.global_variables_initializer())
	# 	batch_x, batch_y = data.getBatch('train', 50)
	# 	result = accuracy.eval(feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
	# 	print(result.shape)
	# exit()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
	

		train_x, train_y = data.getSet('train')
		eval_x, eval_y = data.getSet('eval')

		max_eval = 0
		max_train = 0
		step_num = 4000
		for i in range(step_num):
			batch_x, batch_y = data.getBatch('train', 256)
			
			if i % 100 == 0:
				#evaluate on eval set
				train_accuracy = net.batchEval(train_x, train_y)
				max_train = max(max_train, train_accuracy)
				print('step ', i, 'train accuracy ', train_accuracy)
				eval_accuracy = net.batchEval(eval_x, eval_y)
				max_eval = max(max_eval, eval_accuracy)
				#summary, eval_accuracy = sess.run([net.merged, net.accuracy], feed_dict={net.x: eval_x[:256], net.y: eval_y[:256], net.keep_prob: 1})
				#net.summary_writer.add_summary(summary, i)
				print('step ', i, 'eval accuracy ', eval_accuracy)

			net.trainStep(batch_x, batch_y, keep_prob=0.5)

		print("Maximum eval accuracy ", max_eval)
		print("Maximum train accuracy ", max_train)
		#evaluate on test set
		#test_x, test_y = data.getSet('test')	
		#test_accuracy = net.batchEval(test_x, test_y)
		#print('test accuracy ', test_accuracy) 
main()