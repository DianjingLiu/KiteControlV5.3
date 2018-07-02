#import gym
#from RL_brain import DeepQNetwork

import tensorflow as tf
import numpy as np

class Deep_network(object):
	"""docstring for network"""
	def __init__(self, 
		nn_size,
		learning_rate = 0.05,
		lr_decay_step = 15000,
		lr_decay_rate = 0.9,
		net_name = 'dnn',
		):
		#super(Deep_network, self).__init__()
		self.global_step = tf.Variable(0, trainable=False)
		self.lr = tf.train.exponential_decay(learning_rate, self.global_step, lr_decay_step, lr_decay_rate, staircase=True)
		self.n_layers = len(nn_size)
		self.size = nn_size
		self.build_net(net_name = net_name)
		# When using, restore only network params
		self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=net_name))
		
		# For classification
		self.correct_pred = tf.equal(tf.round(self.pred), self.label)
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred,"float"))
		
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def build_net(self, net_name, activation_function=tf.nn.tanh, init_range = 0.01):
		w_initializer = tf.random_uniform_initializer(-init_range, init_range)
		self.input = tf.placeholder(tf.float32, [None, self.size[0]], name='input')  # input
		self.label = tf.placeholder(tf.float32, [None, self.size[-1]], name='label')  # label
		with tf.variable_scope(net_name):
			output = {}
			output['0'] = self.input
			for j in range(self.n_layers-1):
				output[str(j+1)] = tf.layers.dense(
					inputs=output[str(j)], 
					units=self.size[j+1], 
					kernel_initializer=w_initializer, 
					name='dense_layer'+str(j), 
					activation = activation_function)
			self.pred = 2*output[str(self.n_layers-1)]
		self.loss = tf.reduce_mean(tf.squared_difference(self.pred, self.label))
		#self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
		self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

	def train(self, inputs, label):
		self.sess.run(self._train_op, feed_dict = {self.input: inputs, self.label: label})

	def test(self, inputs, label):
		# For classification problem
		return self.sess.run(self.accuracy, feed_dict = {self.input: inputs, self.label: label})
	
	def predict(self, inputs):
		# For classification problem
		return self.sess.run(self.pred, feed_dict = {self.input: inputs})
	
	def show_loss(self, inputs, label):
		return self.sess.run(self.loss, feed_dict = {self.input: inputs, self.label: label})
	
	def get_lr(self):
		return self.sess.run(self.lr)
	def train_step(self):
		return self.sess.run(self.global_step)
	def reset_train_step(self):
		self.sess.run(self.global_step.assign(0))

	def save(self, filename):
		self.saver.save(self.sess, filename)
	def restore(self, filename):
		self.saver.restore(self.sess, filename)


if __name__ == '__main__':
	trainfile={}
	trainfile = './data/ann-train1.npz'
	testfile =  './data/ann-test1.npz'
	n_input = 21
	n_classes = 1 

	read = np.load(trainfile)
	data = read['dat']
	batch_x = data[:, 0 : n_input]
	batch_y = data[:, n_input : n_input + n_classes]
	
	read = None
	read=np.load(testfile)
	data_test=read['dat']
	test_x = data_test[:, 0: n_input]
	test_y = data_test[:, n_input: n_input + n_classes]
	# set parameters

	nn_size = [n_input, 5000, 500, 200, n_classes]
	data_size = len(data)
	batch_size = data_size
	my_nn = Deep_network(nn_size = nn_size, learning_rate = 0.005)

	import time
	timer=time.clock()
	for i in range(1000):
		#print('=================')
		#print(i)
		#idx = np.random.randint(data_size, size=batch_size)
		#batch_x = data[idx, 0 : n_input]
		#batch_y = data[idx, n_input : n_input + n_classes]
		my_nn.train(inputs=batch_x, label = batch_y)
		#accuracy = my_nn.test(inputs = test_x, label = test_y)
		#print(accuracy)
	print(time.clock()-timer)
	#my_nn.save('testmodel.ckpt')
	"""
	accuracy = my_nn.test(inputs = test_x, label = test_y)
	print(accuracy)
	my_nn.restore('testmodel.ckpt')
	accuracy = my_nn.test(inputs = test_x, label = test_y)
	print(accuracy)
	"""
	"""
	new_nn  = network(nn_size = nn_size, learning_rate = 0., net_name = '1')
	accu = new_nn.test(inputs = test_x, label = test_y)
	print(accu)
	new_nn.restore('testmodel.ckpt')
	accu = new_nn.test(inputs = test_x, label = test_y)
	print(accu)
	"""
	