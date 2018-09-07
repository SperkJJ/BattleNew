"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.99,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        #self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.ep_in1, self.ep_in2, self.ep_in3, self.ep_in4, self.ep_as, self.ep_rs = [], [], [], [], [], []
        self._build_net()
        self.learn_step_counter = 0
        self.merged = tf.summary.merge_all()

        self.saver = tf.train.Saver()

        self.sess = tf.Session()

        self.train_writer = tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        try:
            model_file = tf.train.latest_checkpoint('./new_log')
            self.saver.restore(self.sess, model_file)
        except:
            pass


    @staticmethod
    def max_out(inputs, num_units, axis=None):
        shape = inputs.get_shape().as_list()
        if shape[0] is None:
            shape[0] = -1
        if axis is None:  # Assume that channel is the last dimension
            axis = -1
        num_channels = shape[axis]
        if num_channels % num_units:
            raise ValueError('number of features({}) is not '
                             'a multiple of num_units({})'.format(num_channels, num_units))
        shape[axis] = num_units
        shape += [num_channels // num_units]
        outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keepdims=False)
        return outputs

    @staticmethod
    def make_cnn(inputs, scope_name):
        with tf.variable_scope(scope_name):
            # convolution layer I
            conv_1 = tf.layers.conv2d(
                inputs=inputs,
                filters=512,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                #activation=tf.nn.leaky_relu,
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.glorot_uniform_initializer())

            #maxout_1 = PolicyGradient.max_out(conv_1, 64)

            # max_pooling layer 1
            pool_1 = tf.layers.max_pooling2d(inputs=conv_1, pool_size=[3, 3], strides=2)

            # convolution layer II
            conv_2 = tf.layers.conv2d(
                inputs=pool_1,
                filters=256,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                #activation=tf.nn.leaky_relu,
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.glorot_uniform_initializer())

            #maxout_2 = PolicyGradient.max_out(conv_2, 32)
            # max_pooling layer II
            pool_2 = tf.layers.max_pooling2d(inputs=conv_2, pool_size=[3, 3], strides=2)

            # convolution layer III
            conv_3 = tf.layers.conv2d(inputs=pool_2, filters=128, kernel_size=[3, 3],strides=1, padding="same",
                                      activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.glorot_uniform_initializer())

            #maxout_3 = PolicyGradient.max_out(conv_3, 16)

            # max_pooling layer III
            pool_3 = tf.layers.max_pooling2d(inputs=conv_3, pool_size=[2, 2], strides=2)

            # flat
            in_2 = tf.contrib.layers.flatten(pool_3)

            # aggregate
            #in_2 = tf.concat([in_2, actions], 1)

            # dense layer I
            fc_1 = tf.layers.dense(inputs=in_2, units=128, activation=tf.nn.leaky_relu,
                                   kernel_initializer=tf.glorot_uniform_initializer())
            maxout_4 = PolicyGradient.max_out(fc_1, 8)
            # dropout layer I
            dp_1 = tf.layers.dropout(inputs=maxout_4, rate=0.3)

            # dense layer II
            fc_2 = tf.layers.dense(inputs=dp_1, units=64, activation=tf.nn.leaky_relu,
                                   kernel_initializer=tf.glorot_uniform_initializer())
            maxout_5 = PolicyGradient.max_out(fc_2, 4)
            # dropout layer II
            dp_2 = tf.layers.dropout(inputs=maxout_5, rate=0.3)

            # dense layer III
            out = tf.layers.dense(inputs=dp_2, units=9, activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.glorot_uniform_initializer())

            return out


    @staticmethod
    def deep_nn(input1, input2, input3, input4):
        input1 = tf.reshape(input1, shape=(-1, 12, 12, 3))
        input2 = tf.reshape(input2, shape=(-1, 12, 12, 3))
        input3 = tf.reshape(input3, shape=(-1, 12, 12, 3))
        input4 = tf.reshape(input4, shape=(-1, 12, 12, 3))

        out_1 = PolicyGradient.make_cnn(inputs=input1, scope_name='wall')
        out_2 = PolicyGradient.make_cnn(inputs=input2, scope_name='gas')
        out_3 = PolicyGradient.make_cnn(inputs=input3, scope_name='weapon')
        out_4 = PolicyGradient.make_cnn(inputs=input4, scope_name='path')

        out = tf.concat([out_1, out_2, out_3, out_4], 1)
        fc = tf.layers.dense(inputs=out, units=9, kernel_initializer=tf.random_uniform_initializer())
        dp = tf.layers.dropout(inputs=fc, rate=0.5)
        return dp

    def _build_net(self):
        with tf.name_scope('inputs'):
            #self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
            self.in_1 = tf.placeholder(tf.float32, [None, 12 * 12 * 3], name='wall_input')
            self.in_2 = tf.placeholder(tf.float32, [None, 12 * 12 * 3], name='gas_input')
            self.in_3 = tf.placeholder(tf.float32, [None, 12 * 12 * 3], name='weapon_input')
            self.in_4 = tf.placeholder(tf.float32, [None, 12 * 12 * 3], name='path_input')

        all_act = self.deep_nn(input1=self.in_1, input2=self.in_2, input3=self.in_3, input4=self.in_4)

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability


        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            #neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            #neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            neg_log_prob = tf.reduce_sum(-tf.log(tf.clip_by_value(self.all_act_prob,1e-10,1.0))*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('entropy'):
            entropy = tf.reduce_mean(tf.reduce_sum(-tf.log(self.all_act_prob) * self.all_act_prob, axis=1))
        with tf.name_scope('data'):
            self.reward = tf.placeholder(tf.float32)
            tf.summary.scalar('reward', self.reward)
            tf.summary.scalar('loss_output', loss)
            tf.summary.scalar('actions_entropy', entropy)
        with tf.name_scope('paras'):
            paras = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='deep_nn')
            for para in paras:
                tf.summary.histogram(para.name, para)
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr, epsilon=1.0).minimize(loss)


    def choose_action(self, observation):
        in_ = observation[:-8]
        # actions = inputs[:, -8:]
        in_1 = in_[-12 * 12 * 3:]
        in_2 = in_[-2 * 12 * 12 * 3:-12 * 12 * 3]
        in_3 = in_[-3 * 12 * 12 * 3:-2 * 12 * 12 * 3]
        in_4 = in_[0:12 * 12 * 3]
        # in_ = tf.reshape(in_, (-1, 12, 12, 6))  # 毒气，墙壁，道具，自身位置，敌方位置，历史路径

        in_1 = np.array(in_1).reshape(1, 12 * 12 * 3)
        in_2 = np.array(in_2).reshape(1, 12 * 12 * 3)
        in_3 = np.array(in_3).reshape(1, 12 * 12 * 3)
        in_4 = np.array(in_4).reshape(1, 12 * 12 * 3)

        #prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        prob_weights = self.sess.run(self.all_act_prob, feed_dict=
        {self.in_1: in_1, self.in_2: in_2, self.in_3: in_3, self.in_4: in_4})
        # select action w.r.t the actions prob
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        #action = np.random.choice(prob_weights.shape[1], p=prob_weights.ravel())
        #action = np.argmax(prob_weights)
        return action

    def store_transition(self, s, a, r):
        #self.ep_obs.append(s)
        s = s[:-8]
        s = s.reshape(4, 12 * 12 * 3)
        self.ep_in1.append(s[0, :])
        self.ep_in2.append(s[1, :])
        self.ep_in3.append(s[2, :])
        self.ep_in4.append(s[3, :])
        self.ep_as.append(a)
        self.ep_rs.append(float(r))

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        _, summary = self.sess.run((self.train_op, self.merged), feed_dict={
             self.in_1: self.ep_in1,
             self.in_2: self.ep_in2,
             self.in_3: self.ep_in3,
             self.in_4: self.ep_in4,
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
             self.reward: sum(self.ep_rs)
        })
        #self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        self.ep_in1, self.ep_in2, self.ep_in3, self.ep_in4, self.ep_as, self.ep_rs = [], [], [], [], [], []
        if self.learn_step_counter % 200 == 0:
            self.train_writer.add_summary(summary, self.learn_step_counter)
            self.saver.save(self.sess, 'new_log/my_model', global_step=self.learn_step_counter)
            print('step{}, running info saved!'.format(self.learn_step_counter))
        self.learn_step_counter += 1
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs, dtype=np.float32)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        if len(discounted_ep_rs) > 1:
            discounted_ep_rs -= np.mean(discounted_ep_rs)
            discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

