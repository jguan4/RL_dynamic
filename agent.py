import sys, re
sys.dont_write_bytecode = True

import gym
import numpy as np
import tensorflow as tf
import random
import os
import subprocess
import replay_memory as rplm
import utils
import time
from tensorflow.python.saved_model import tag_constants

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


#####################################  Description  ####################################################
# This file defines the class DQN_Agent. It uses Double DQN to train the neural network.
########################################################################################################

class DQN_Agent:


	# Description: Initializes the DQN_Agent object
	# Parameters:
	# - environment: 		Object supporting methods like 'reset', 'render', 'step' etc.
	# 			     		For an example see environment.py.
	# - architecture: 		Object supporting the method 'evaluate', usually a neural network. 
	# 				  		for exapmles see parameters/architectures.py.
	# - explore_rate: 		Object supporting the method 'get'. See parameters/explore_rates.py for examples.
	# - learning_rate: 		Object supporting the method 'get'. See parameters/learning_rates.py for examples.
	# - batch_size: 		Integer giving the size of the minibatch to be used in the optimization process.
	# - memory_capacity: 	Integer giving the size of the replay memory.
	# - num_episodes: 		Integer specifying the number of training episodes to be used. 
	# - learning_rate_drop_frame_limit:
	# 						Integer specifying by which frame during training the minimal explore rate should be reached 
	# - target_update_frequency:
	# 						Integer specifying the frequency at which the target network's weights are to be updated
	# - discount:			Number in (0,1) giving the discount factor
	# - delta:  			Number, the delta parameter in huber loss
	# - model_name:  		String, the name of the folder where the model is saved. 
	# Output: None

    def __init__(self, environment, architecture, explore_rate, learning_rate,
                 batch_size, memory_capacity, num_episodes, learning_rate_drop_frame_limit,
                 target_update_frequency, discount=0.99, delta=1, model_name=None):
        self.env = environment
        self.architecture = architecture()
        self.explore_rate = explore_rate()
        self.learning_rate = learning_rate()
        self.model_path = os.path.dirname(os.path.realpath(__file__)) + '/models/' + model_name if model_name else str(self.env)
        self.log_path = self.model_path + '/log'
        self.initialize_tf_variables()

        # Training parameters setup
        self.target_update_frequency = target_update_frequency
        self.discount = discount
        self.best_training_score = None;
        self.replay_memory = rplm.Replay_Memory(memory_capacity, batch_size)
        # self.training_metadata = utils.Training_Metadata(frame=self.sess.run(self.frames), frame_limit=learning_rate_drop_frame_limit,
        # 												   episode=self.sess.run(self.episode), num_episodes=num_episodes)
        self.training_metadata = utils.Training_Metadata(frame=0, frame_limit=learning_rate_drop_frame_limit,
                                                         episode=0, num_episodes=num_episodes)
        self.delta = delta
        utils.document_parameters(self)

    # Description: Sets up tensorflow graph and other variables, only called internally
    # Parameters: None
    # Output: None
    def initialize_tf_variables(self):
        # Setting up game specific variables
        self.state_size = self.env.state_space_size
        self.action_size = self.env.action_space_size
        self.state_shape = self.env.state_shape
        self.q_grid = None

        # Tf placeholders
        self.state_tf = tf.placeholder(shape=self.state_shape, dtype=tf.float32, name='state_tf')
        self.action_tf = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32, name='action_tf')
        self.y_tf = tf.placeholder(dtype=tf.float32, name='y_tf')
        self.alpha = tf.placeholder(dtype=tf.float32, name='alpha')
        self.test_score = tf.placeholder(dtype=tf.float32, name='test_score')
        self.avg_q = tf.placeholder(dtype=tf.float32, name='avg_q')
        self.loss_value = tf.placeholder(dtype=tf.float32, name='loss_value')
        self.henon_x1 = tf.placeholder(dtype=tf.float32, name='henon_x1')
        self.henon_x2 = tf.placeholder(dtype=tf.float32, name='henon_x2')
        self.scat_1 = tf.placeholder(dtype=tf.float32, name='scat_1')
        self.scat_2 = tf.placeholder(dtype=tf.float32, name='scat_2')
        self.f_count = tf.placeholder(dtype=tf.float32, name='f_count')

        # Keep track of episode and frames
        self.episode = tf.Variable(initial_value=0, trainable=False, name='episode')
        self.frames = tf.Variable(initial_value=0, trainable=False, name='frames')
        self.increment_frames_op = tf.assign(self.frames, self.frames + 1, name='increment_frames_op')
        self.increment_episode_op = tf.assign(self.episode, self.episode + 1, name='increment_episode_op')

        # Operations
        # NAME                      DESCRIPTION                                         FEED DEPENDENCIES
        # Q_value                   Value of Q at given state(s)                        state_tf
        # Q_argmax                  Action(s) maximizing Q at given state(s)            state_tf
        # Q_amax                    Maximal action value(s) at given state(s)           state_tf
        # Q_value_at_action         Q value at specific (action, state) pair(s)         state_tf, action_tf
        # onehot_greedy_action      One-hot encodes greedy action(s) at given state(s)  state_tf
        self.Q_value = self.architecture.evaluate(self.state_tf, self.action_size)
        self.Q_argmax = tf.argmax(self.Q_value, axis=1, name='Q_argmax')
        self.Q_amax = tf.reduce_max(self.Q_value, axis=1, name='Q_max')
        self.Q_value_at_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_tf), axis=1, name='Q_value_at_action')
        self.onehot_greedy_action = tf.one_hot(self.Q_argmax, depth=self.action_size)

        # Training related
        # NAME                          FEED DEPENDENCIES
        # loss                          y_tf, state_tf, action_tf
        # train_op                      y_tf, state_tf, action_tf, alpha

        # self.loss = tf.losses.mean_squared_error(self.y_tf, self.Q_value_at_action)
        self.loss = tf.losses.huber_loss(self.y_tf, self.Q_value_at_action)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha)
        self.train_op = self.optimizer.minimize(self.loss, name='train_minimize')

        # Tensorflow session setup
        self.saver = tf.train.Saver(max_to_keep=None)
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)
        self.trainable_variables = tf.trainable_variables()

        # Tensorboard setup
        self.writer = tf.summary.FileWriter(self.log_path)
        self.writer.add_graph(self.sess.graph)
        test_score = tf.summary.scalar("Training score", self.test_score, collections=None, family=None)
        avg_q = tf.summary.scalar("Average Q-value", self.avg_q, collections=None, family=None)
        loss = tf.summary.scalar("Loss", self.loss_value, collections=None, family=None)
        henon_x1 = tf.summary.scalar("henon_x1", self.henon_x1, collections=None, family=None)
        henon_x2 = tf.summary.scalar("henon_x2", self.henon_x2, collections=None, family=None)
        scat_1 = tf.summary.scalar("scat_1", self.scat_1, collections=None, family=None)
        scat_2 = tf.summary.scalar("scat_2", self.scat_2, collections=None, family=None)
        f_count= tf.summary.scalar("f_count", self.f_count, collections=None, family=None)
        self.training_summary = tf.summary.merge([avg_q])
        self.update_summary = tf.summary.merge([loss])
        self.test_summary = tf.summary.merge([test_score])
        self.traj_summary = tf.summary.merge([henon_x1,henon_x2])
        self.scat_summary = tf.summary.merge([scat_1,scat_2])
        self.frame_summary = tf.summary.merge([f_count])
        # subprocess.Popen(['tensorboard', '--logdir', self.log_path])

        # Initialising variables and finalising graph
        self.sess.run(tf.global_variables_initializer())
        self.fixed_target_weights = self.sess.run(self.trainable_variables)

        self.sess.graph.finalize()

    # Description: Performs one step of batch gradient descent on the DDQN loss function. 
    # Parameters:
    # - alpha: Number, the learning rate 
    # Output: None
    def experience_replay(self, alpha):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_memory.get_mini_batch(self.training_metadata)
        y_batch = [None] * self.replay_memory.batch_size
        fixed_feed_dict = {self.state_tf: next_state_batch}
        fixed_feed_dict.update(zip(self.trainable_variables, self.fixed_target_weights))
        greedy_actions = self.sess.run(self.onehot_greedy_action, feed_dict={self.state_tf: next_state_batch})
        fixed_feed_dict.update({self.action_tf: greedy_actions})
        Q_batch = self.sess.run(self.Q_value_at_action, feed_dict=fixed_feed_dict)
        y_batch = reward_batch + self.discount * np.multiply(np.invert(done_batch), Q_batch)

        loss_value = self.sess.run(self.loss, feed_dict={self.y_tf:y_batch, self.Q_value_at_action:Q_batch})
        self.writer.add_summary(self.sess.run(self.update_summary, feed_dict={self.loss_value: loss_value}), self.training_metadata.frame)

        feed = {self.state_tf: state_batch, self.action_tf: action_batch, self.y_tf: y_batch, self.alpha: alpha}
        self.sess.run(self.train_op, feed_dict=feed)

    # Description: Chooses action wrt an e-greedy policy. 
    # Parameters:
    # - state: 		Tensor representing a single state
    # - epsilon: 	Number in (0,1)
    # Output: 		Integer in the range 0...self.action_size-1 representing an action
    def get_action(self, state, epsilon):
        # Performing epsilon-greedy action selection
        if random.random() < epsilon:
            return self.env.sample_action_space()
        else:
            return self.sess.run(self.Q_argmax, feed_dict={self.state_tf: [state]})[0]

    # Description: Updates the weights of the target network
    # Parameters: 	None
    # Output: 		None
    def update_fixed_target_weights(self):
        self.fixed_target_weights = self.sess.run(self.trainable_variables)

    # Description: Trains the model
    # Parameters: 	None
    # Output: 		None
    def train(self):
        while self.sess.run(self.episode) < self.training_metadata.num_episodes:
            episode = self.sess.run(self.episode)
            self.training_metadata.increment_episode()
            self.sess.run(self.increment_episode_op)

            # Setting up game environment
            state = self.env.reset()
            self.env.render()

            # Setting up parameters for the episode
            done = False
            episode_frame = 0
            while not done:
                epsilon = self.explore_rate.get(self.training_metadata)
                alpha = self.learning_rate.get(self.training_metadata)

                # self.env.render()
                # Updating fixed target weights every #target_update_frequency frames
                if self.training_metadata.frame % self.target_update_frequency == 0 and (self.training_metadata.frame != 0):
                    self.update_fixed_target_weights()

                # Choosing and performing action and updating the replay memory
                action = self.get_action(state, epsilon)
                next_state, reward, done, info = self.env.step(action)
                episode_frame += 1
                print("Frame {0} \t Reward: {1} \t State: {2} \t Radius: {3}".format(self.training_metadata.frame, reward, next_state, info['Radius']))

                # print("State: {0} \t Reward: {1}".format(next_state,reward))
                # if reward == 1:
                #     print(info['Fixed_Point'])
                #     utils.pause()

                if isinstance(info['Fixed_Point'], np.ndarray):
                    fp = info['Fixed_Point']
                    self.writer.add_summary(self.sess.run(self.traj_summary,
                        feed_dict={self.henon_x1: fp[0], self.henon_x2: fp[1]}), self.training_metadata.frame)
                self.writer.add_summary(self.sess.run(self.scat_summary,
                    feed_dict={self.scat_1: next_state[0], self.scat_2: next_state[1]}), self.training_metadata.frame)
                score = info['Consecutive_Reward']

                self.replay_memory.add(self, state, action, reward, next_state, done)

                # Performing experience replay if replay memory populated
                if self.replay_memory.length() > self.replay_memory.batch_size: #100 * self.replay_memory.batch_size:
                    self.sess.run(self.increment_frames_op)
                    self.training_metadata.increment_frame()
                    self.experience_replay(alpha)
                state = next_state
                done = info['true_done']

                # Creating q_grid if not yet defined and calculating average q-value
                if self.replay_memory.length() > 1000:
                    self.q_grid = self.replay_memory.get_q_grid(size=200, training_metadata=self.training_metadata)
                avg_q = self.estimate_avg_q()
                self.writer.add_summary(self.sess.run(self.training_summary, feed_dict={self.avg_q: avg_q}), self.training_metadata.frame)

                # if (self.training_metadata.frame % 3000 == 0) and (self.training_metadata.frame != 0):
                #     score, std, rewards = self.test(num_test_episodes=5, visualize=True)
                #     if self.best_training_score==None or score>self.best_training_score:
                #         self.best_training_score = score
                #         self.delete_previous_checkpoints()
                #         self.saver.save(self.sess, self.model_path + '/best.data.chkp', global_step=self.training_metadata.episode)
                #     # if (self.training_metadata.num_episodes - episode)<30:
                #     #     self.saver.save(self.sess, self.model_path + '/last.data.chkp', global_step=self.training_metadata.episode)
                #     print('{0} +- {1}'.format(score, std))
                #     self.writer.add_summary(self.sess.run(self.test_summary,
                #                                           feed_dict={self.test_score: score}), self.training_metadata.frame)

            if self.best_training_score==None or episode_frame<self.best_training_score:#score>self.best_training_score:
                self.best_training_score = episode_frame
                self.delete_previous_checkpoints()
                self.saver.save(self.sess, self.model_path + '/best.data.chkp', global_step=self.training_metadata.episode)
            self.writer.add_summary(self.sess.run(self.test_summary,
                feed_dict={self.test_score: episode_frame}), self.training_metadata.episode)
            self.writer.add_summary(self.sess.run(self.frame_summary,
                feed_dict={self.f_count: self.training_metadata.frame}), self.training_metadata.episode)

            # Saving tensorboard data and model weights
            # if (self.training_metadata.frame % 300 == 0) and (self.training_metadata.frame != 0):
            #     score, std, rewards = self.test(num_test_episodes=5, visualize=True)
            #     if self.best_training_score==None or score>self.best_training_score:
            #         self.best_training_score = score
            #         self.delete_previous_checkpoints()
            #         self.saver.save(self.sess, self.model_path + '/best.data.chkp', global_step=self.training_metadata.episode)
            #     # if (self.training_metadata.num_episodes - episode)<30:
            #     #     self.saver.save(self.sess, self.model_path + '/last.data.chkp', global_step=self.training_metadata.episode)
            #     print('{0} +- {1}'.format(score, std))
            #     self.writer.add_summary(self.sess.run(self.test_summary,
            #                                           feed_dict={self.test_score: score}), self.training_metadata.frame)
                
            # self.writer.add_summary(self.sess.run(self.training_summary, feed_dict={self.avg_q: avg_q}), self.training_metadata.frame)

    # Description: Tests the model
    # Parameters:
    # - num_test_episodes: 	Integer, giving the number of episodes to be tested over
    # - visualize: 			Boolean, gives whether should render the testing gameplay
    def test(self, num_test_episodes, visualize, pause=False):
        rewards = []
        for episode in range(num_test_episodes):
            done = False
            state = self.env.reset(test=True)
            episode_reward = 0
            reward = 0
            frame = 0
            if not visualize:
                self.test_env.render()
            # while not done:
            while frame < 1000:
                if visualize:
                    self.env.render()
                action = self.get_action(state, epsilon=0)
                next_state, reward, done, info = self.env.step(action, test=True)
                frame += 1
                state = next_state
                print("Reward: {0} \t State: {1} \t Fixed Point: {2}".format(reward, state, info['Fixed_Point']))
                episode_reward += reward
                done = info['true_done']
                if pause: utils.pause()
            rewards.append(episode_reward)

        return np.mean(rewards), np.std(rewards), rewards

    # Description: Returns average Q-value over some number of fixed tracks
    # Parameters: 	None
    # Output: 		None
    def estimate_avg_q(self):
        if not self.q_grid:
            return 0
        return np.average(np.amax(self.sess.run(self.Q_value, feed_dict={self.state_tf: self.q_grid}), axis=1))

    # Description: Loads a model trained in a previous session
    # Parameters:
    # - path: 	String, giving the path to the checkpoint file to be loaded
    # Output:	None
    def load(self, path):
        self.saver.restore(self.sess, path)

    def delete_previous_checkpoints(self):
        my_dir = self.model_path
        for fname in os.listdir(my_dir):
            if re.match("best.data", fname):
                os.remove(os.path.join(my_dir, fname))
