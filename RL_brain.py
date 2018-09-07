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

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

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
    def deep_nn(inputs):
        in_ = inputs[:, :-8]
        actions = inputs[:, -8:]

        in_ = tf.reshape(in_, (-1, 12, 12, 6))  # 毒气，墙壁，道具，自身位置，敌方位置，历史路径
        # conv1
        conv_1 = tf.layers.conv2d(
            inputs=in_,
            filters=256,
            kernel_size=[3, 3],
            strides=1,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.uniform_unit_scaling_initializer())
        tf.summary.histogram('convolution layer I', conv_1)
        # conv2
        conv_2 = tf.layers.conv2d(
            inputs=conv_1,
            filters=128,
            kernel_size=[3, 3],
            strides=1,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.uniform_unit_scaling_initializer())
        tf.summary.histogram('convolution layer II', conv_2)
        # max_pooling1
        pool_1 = tf.layers.max_pooling2d(inputs=conv_2, pool_size=[3, 3], strides=2)

        # conv3
        conv_3 = tf.layers.conv2d(
            inputs=pool_1,
            filters=64,
            kernel_size=[3, 3],
            strides=1,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.uniform_unit_scaling_initializer())
        tf.summary.histogram('convolution layer III', conv_3)
        # max_pooling2
        pool_2 = tf.layers.max_pooling2d(inputs=conv_3, pool_size=[2, 2], strides=2)

        # flat
        in_2 = tf.contrib.layers.flatten(pool_2)

        # aggregate
        in_2 = tf.concat([in_2, actions], 1)

        # dense1
        fc_1 = tf.layers.dense(inputs=in_2, units=512, activation=tf.nn.relu,
                              kernel_initializer=tf.uniform_unit_scaling_initializer())
        tf.summary.histogram('fully connected layer I', fc_1)
        # dropout1
        dp_1 = tf.layers.dropout(inputs=fc_1, rate=0.3)

        # dense2
        fc_2 = tf.layers.dense(inputs=dp_1, units=256, activation=tf.nn.relu,
                              kernel_initializer=tf.uniform_unit_scaling_initializer())
        tf.summary.histogram('fully connected layer II', fc_2)
        # dropout2
        dp_2 = tf.layers.dropout(inputs=fc_2, rate=0.3)

        # out
        out = tf.layers.dense(inputs=dp_2, units=9, kernel_initializer=tf.uniform_unit_scaling_initializer())
        tf.summary.histogram('deep nn output', out)

        return out

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        all_act = self.deep_nn(inputs=self.tf_obs)
        tf.summary.histogram('all_act', all_act)
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            #neg_log_prob = tf.reduce_sum(-tf.log(tf.clip_by_value(self.all_act_prob, 1e-10, 1.0))
            #                             * tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('entropy'):
            entropy =  tf.reduce_mean(tf.reduce_sum(-tf.log(self.all_act_prob) * self.all_act_prob, axis=1))

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
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        _, summary = self.sess.run((self.train_op, self.merged), feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
             self.reward: sum(self.ep_rs)
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data

        if self.learn_step_counter % 200 == 0:
            self.train_writer.add_summary(summary, self.learn_step_counter)
            self.saver.save(self.sess, 'new_log/my_test_model', global_step=self.learn_step_counter)
            print(f'step {self.learn_step_counter}, running info saved.')

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
