"""
Using Tensorflow to build the neural network.
Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
import cv2
from pdb import set_trace

np.random.seed(1)
tf.set_random_seed(1)

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_input,
            n_hid=[200, 100, 100, 50, 20],
            learning_rate=0.01,
            lr_decay_step = 15000,
            lr_decay_rate = 0.9,
            reward_decay=0.99,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_max=50000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_input = n_input # figure size [n1, n2, n_channel]
        self.n_hid = n_hid

        # total learning step
        self.global_step = tf.Variable(0, trainable=False)

        # exponential decay learning rate
        #self.lr = learning_rate
        self.lr = tf.train.exponential_decay(learning_rate, self.global_step, lr_decay_step, lr_decay_rate, staircase=True)

        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_max = memory_max
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # consist of [target_net, evaluate_net]
        self._build_net()
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net_params'))

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        '''   
        # build CNN network 
        self.s  = tf.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], self.n_input[2]], name='s')  # state, input for eval net
        self.s_ = tf.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], self.n_input[2]], name='s_') # new state, input for target net
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        self.q_eval = self.build(self.s, 'eval_net_params')      
        self.q_next = self.build(self.s_, 'target_net_params')   # [None, n_actions]
        '''
        # build fully connected network 
        self.s  = tf.placeholder(tf.float32, [None, self.n_input], name='s')  # state, input for eval net
        self.s_ = tf.placeholder(tf.float32, [None, self.n_input], name='s_') # new state, input for target net
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        self.q_eval = self.build_dense(self.s,  self.n_hid, 'eval_net_params')      
        self.q_next = self.build_dense(self.s_, self.n_hid, 'target_net_params')   # [None, n_actions]
     
        
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net_params')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            #self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss, var_list=e_params, global_step=self.global_step)
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=e_params, global_step=self.global_step)
            
    def build(self, input_layer, name):
        with tf.variable_scope(name):
            with tf.variable_scope('conv'):
                conv1 = tf.layers.conv2d(
                    inputs=input_layer,
                    filters=32,
                    kernel_size=[8, 8],
                    padding="same",
                    activation=tf.nn.relu,
                    name='c1')
                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=4)
                conv2 = tf.layers.conv2d(
                    inputs=input_layer,
                    filters=64,
                    kernel_size=[4, 4],
                    padding="same",
                    activation=tf.nn.relu,
                    name='c2')
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
                length=int(pool2.shape[1]*pool2.shape[2]*pool2.shape[3])
                pool2_flat = tf.reshape(pool2, [-1, length])
            with tf.variable_scope('dense'):
                d1=tf.layers.dense(inputs=pool2_flat, 
                    units=512,#1024, 
                    name='d1', 
                    activation = tf.nn.relu)
                out=tf.layers.dense(inputs=d1, 
                    units=self.n_actions, 
                    name='out', 
                    #activation = None
                    )
        return out
    """
    def build_dense(self, input_layer, name):
        # initialize the weights so that initial Q values are close to 0
        initializer = tf.random_uniform_initializer(-0.0001, 0.0001) 
        # Regularize weights and biases
        regularizer = tf.contrib.layers.l2_regularizer(0.01, scope=None)
        with tf.variable_scope(name):
            d1=tf.layers.dense(inputs=input_layer, 
                units=200,#1024, 
                kernel_initializer=initializer, bias_initializer=initializer,
                kernel_regularizer=regularizer, bias_regularizer=regularizer,
                #name='d1', 
                activation = tf.nn.tanh)
            d2=tf.layers.dense(inputs=d1, 
                units=100,#1024, 
                kernel_initializer=initializer, bias_initializer=initializer,
                kernel_regularizer=regularizer, bias_regularizer=regularizer,
                #name='d2', 
                activation = tf.nn.tanh)
            d3=tf.layers.dense(inputs=d2, 
                units=50,#1024, 
                kernel_initializer=initializer, bias_initializer=initializer,
                kernel_regularizer=regularizer, bias_regularizer=regularizer,
                #name='d2', 
                activation = tf.nn.tanh)
            out=tf.layers.dense(inputs=d3, 
                units=self.n_actions, 
                kernel_initializer=initializer, bias_initializer=initializer,
                kernel_regularizer=regularizer, bias_regularizer=regularizer,
                #name='out', 
                #activation = None
                )
        return out
    """
    def build_dense(self, input_layer, n_hid, name):
        # initialize the weights so that initial Q values are close to 0
        initializer = tf.random_uniform_initializer(-0.0001, 0.0001) 
        # Regularize weights and biases
        regularizer = tf.contrib.layers.l2_regularizer(0.01, scope=None)
        with tf.variable_scope(name):
            hid = input_layer
            for n in n_hid:
                hid = tf.layers.dense(inputs=hid, 
                            units=n,
                            kernel_initializer=initializer, bias_initializer=initializer,
                            kernel_regularizer=regularizer, bias_regularizer=regularizer,
                            activation = tf.nn.tanh)
            out = tf.layers.dense(inputs=hid, 
                        units=self.n_actions, 
                        kernel_initializer=initializer, bias_initializer=initializer,
                        kernel_regularizer=regularizer, bias_regularizer=regularizer,
                        #activation = None
                        )
        return out


    def store_transition(self, s, a, r, s_, done=False):
        a = a[np.newaxis,:]  if a.ndim==1  else a
        r = r[np.newaxis,:]  if r.ndim==1  else r
        s = s[np.newaxis,:]  if s.ndim==1  else s
        s_= s_[np.newaxis,:] if s_.ndim==1 else s_
        done = done[np.newaxis,:]  if done.ndim==1  else done
        if not hasattr(self, 'memory_a'):
            self.memory_a = a  # action. shape=[None, 1]
            self.memory_r = r  # reward. shape=[None, 1]
            self.memory_s = s  # state.  shape=[None, n_f1, n_f2, n_channel]
            self.memory_s_= s_ # new state. shape=[None, n_f1, n_f2, n_channel]
            self.memory_d = done  # game end label. shape=[None, 1]. done=True means game over
        else:
            self.memory_a = np.concatenate((self.memory_a, a), axis=0)
            self.memory_r = np.concatenate((self.memory_r, r), axis=0)
            self.memory_s = np.concatenate((self.memory_s, s), axis=0)
            self.memory_s_= np.concatenate((self.memory_s_, s_), axis=0)
            self.memory_d = np.concatenate((self.memory_d, done), axis=0)

        if len(self.memory_a) > self.memory_max:
            n_delete = len(self.memory_a) - self.memory_max
            self.memory_a = np.delete(self.memory_a, range(n_delete), 0)
            self.memory_r = np.delete(self.memory_r, range(n_delete), 0)
            self.memory_s = np.delete(self.memory_s, range(n_delete), 0)
            self.memory_s_= np.delete(self.memory_s_, range(n_delete), 0)
            self.memory_d = np.delete(self.memory_d, range(n_delete), 0)
        self.memory_size = len(self.memory_a)
        '''
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        '''
    def load_memory(self, filename):
        read = np.load(filename,encoding='latin1')
        self.memory_a  = read['memory_a']
        self.memory_r  = read['memory_r']
        self.memory_s  = read['memory_s']
        self.memory_s_ = read['memory_s1']
        self.memory_d  = read['memory_d']
        self.memory_size = len(self.memory_a)

    def choose_action(self, observation):
        # Add batch dimension when feed into tf placeholder
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.sess.run(self.global_step) % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            #print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        sample_index = np.random.choice(self.memory_size, size=self.batch_size)

        # calculate target q value
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: self.memory_s_[sample_index],  # fixed params
                self.s:  self.memory_s[sample_index],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        eval_act_index = self.memory_a[sample_index].astype(int).reshape((self.batch_size))
        reward = self.memory_r[sample_index]

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        #not_terminate = (reward == 0)
        not_terminate = 1-self.memory_d[sample_index]
        q_target[batch_index, eval_act_index] = (reward + self.gamma * np.amax(q_next, axis=1, keepdims=True) * not_terminate).T

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]
        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]
        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: self.memory_s[sample_index],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        #self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
    def get_lr(self):
        return self.sess.run(self.lr)
    def train_step(self):
        return self.sess.run(self.global_step)
    def reset_train_step(self):
        self.sess.run(self.global_step.assign(0))
    def renew_epsilon(self):
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

    def plot_cost(self, filename=None):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def save(self, filename):
        self.saver.save(self.sess, filename)

    def restore(self, filename):
        self.saver.restore(self.sess, filename)

    # for debug
    def show_q(self, observation):
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        return actions_value

class Memory:
    def __init__(self, max_size):
        self.size = 0
        self.max_size = max_size

    def store(self, s, a, r, s_):
        if not hasattr(self, 'a'):
            self.a = a  # action. shape=[None, 1]
            self.r = r  # reward. shape=[None, 1]
            self.s = s  # state.  shape=[None, n_f1, n_f2, n_channel]
            self.s_= s_ # new state. shape=[None, n_f1, n_f2, n_channel]
        else:
            self.a = np.concatenate((self.a, a), axis=0)
            self.r = np.concatenate((self.r, r), axis=0)
            self.s = np.concatenate((self.s, s), axis=0)
            self.s_= np.concatenate((self.s_, s_), axis=0)
        if len(self.a) > self.max:
            n_delete = len(self.a) - self.max_size
            self.a = np.delete(self.a, range(n_delete), 0)
            self.r = np.delete(self.r, range(n_delete), 0)
            self.s = np.delete(self.s, range(n_delete), 0)
            self.s_= np.delete(self.s_, range(n_delete), 0)
        self.size = len(self.a)
    def sample(self, batch_size):
        idx = np.random.choice(self.size, size=batch_size)
        return self.s[idx], self.s_[idx], self.a[idx], self.r[idx]

# processing image input
class processor:
    def __init__(self, 
        #size, # size of the RGB image
        new_size = [84, 84], # size of the converted state [n1, n2]
        m=4,
        ):
        #self.size = size 
        self.new_size = new_size 
        self.m = m
    def add(self, images):
        gray_image = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
        pr_img = cv2.resize(gray_image,(self.new_size[0],self.new_size[1]))
        if not hasattr(self, 'state'):
            self.state = np.copy(pr_img).reshape((1,pr_img.shape[0],pr_img.shape[1],1))
            return None
        else:
            self.state = np.concatenate((self.state, pr_img.reshape((1,pr_img.shape[0],pr_img.shape[1],1))),axis=3)
            if self.state.shape[3]>self.m:
                self.state = np.delete(self.state, 0, axis=3)
                return self.state
            else:
                return None


