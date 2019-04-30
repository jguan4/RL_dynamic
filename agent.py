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
import pickle
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
                 target_update_frequency, discount=0.9, delta=1, model_name=None):
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
        self.action_shape = self.env.action_shape
        self.output_size = 1
        self.q_grid = None

        # Tf placeholders
        self.state_tf = tf.placeholder(shape=self.state_shape, dtype=tf.float32, name='state_tf')
        # self.action_chosen = tf.placeholder(shape=self.action_shape, dtype=tf.float32, name='action_chosen')
        self.action_tf = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32, name='action_tf')
        self.y_tf = tf.placeholder(dtype=tf.float32, name='y_tf')
        self.alpha = tf.placeholder(dtype=tf.float32, name='alpha')

        # for tensorboard log
        self.test_score = tf.placeholder(dtype=tf.float32, name='test_score')
        self.avg_q = tf.placeholder(dtype=tf.float32, name='avg_q')
        self.loss_value = tf.placeholder(dtype=tf.float32, name='loss_value')
        # self.P_x1 = tf.placeholder(dtype=tf.float32, name='P_x1')
        # self.henon_x2 = tf.placeholder(dtype=tf.float32, name='henon_x2')
        # self.scat_1 = tf.placeholder(dtype=tf.float32, name='scat_1')
        # self.scat_2 = tf.placeholder(dtype=tf.float32, name='scat_2')
        # self.fp = tf.placeholder(dtype=tf.float32, shape=self.state_shape, name='fp')
        self.traj = tf.placeholder(dtype=tf.float32, name='traj')
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

        # Input includes action
        # self.Q_value = self.architecture.evaluate(tf.concat([self.state_tf,self.action_chosen], 1), self.output_size)

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
        # henon_x2 = tf.summary.scalar("henon_x2", self.henon_x2, collections=None, family=None)
        # P_x1 = tf.summary.scalar("P_x1", self.P_x1, collections=None, family=None)
        # scat_1 = tf.summary.scalar("scat_1", self.scat_1, collections=None, family=None)
        # scat_2 = tf.summary.scalar("scat_2", self.scat_2, collections=None, family=None)
        # fp = tf.summary.histogram("fp",self.fp,collections=None, family=None)
        traj = tf.summary.scalar("traj", self.traj, collections=None, family=None)
        f_count= tf.summary.scalar("f_count", self.f_count, collections=None, family=None)
        self.training_summary = tf.summary.merge([avg_q])
        self.update_summary = tf.summary.merge([loss])
        self.test_summary = tf.summary.merge([test_score])
        # self.traj_summary = tf.summary.merge([P_x1])
        self.scat_summary = tf.summary.merge([traj])
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
        # fixed_feed_dict.update({self.action_chosen: action_batch})
        # Q_batch = self.sess.run(self.Q_value, feed_dict=fixed_feed_dict)
        Q_batch = self.sess.run(self.Q_value_at_action, feed_dict=fixed_feed_dict)
        y_batch = reward_batch + self.discount * np.multiply(np.invert(done_batch), Q_batch)

        loss_value = self.sess.run(self.loss, feed_dict={self.y_tf:y_batch, self.Q_value_at_action:Q_batch})
        self.writer.add_summary(self.sess.run(self.update_summary, feed_dict={self.loss_value: loss_value}), self.training_metadata.frame)

        feed = {self.state_tf: state_batch, self.action_tf: action_batch, self.y_tf: y_batch, self.alpha: alpha}
        # feed = {self.state_tf: state_batch, self.action_chosen: action_batch, self.y_tf: y_batch, self.alpha: alpha}
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
            if self.state_shape[0]:
                feed_in = state
            else: 
                feed_in = [state]
            return self.sess.run(self.Q_argmax, feed_dict={self.state_tf: feed_in})[0]
            # states_repeat = np.tile(state,(self.action_size,1))
            # actions_possible = np.arange(self.action_size).reshape((self.action_size, 1))
            # return self.sess.run(self.Q_argmax, 
                # feed_dict={self.state_tf: states_repeat,self.action_chosen: actions_possible})[0]


    # Description: Updates the weights of the target network
    # Parameters: 	None
    # Output: 		None
    def update_fixed_target_weights(self):
        self.fixed_target_weights = self.sess.run(self.trainable_variables)

    def fill_random(self):
        state = self.env.reset()
        eps = 1000
        done = False
        for i in range(eps):
            if done:
                state = self.env.reset()
                done = False
            while not done:
                action = self.get_action(state, 1)
                next_state, reward, done, info = self.env.step(action)
                self.replay_memory.add(self, state, action, reward, next_state, done)
                state = next_state


    # Description: Trains the model
    # Parameters: 	None
    # Output: 		None
    def train(self):
        # training_scores = np.empty((0,2),float)
        frame_eps = np.empty((0,2),float)
        # traj = np.empty((0,1),float)
        # period_points = np.empty((0,self.state_size),float)

        # self.fill_random()
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
            update = False

            freq = self.replay_memory.get_replay_frequency(self.training_metadata)
            # while episode_frame<1000:
            while not done:
                epsilon = self.explore_rate.get(self.training_metadata)
                alpha = self.learning_rate.get(self.training_metadata)

                # self.env.render()
                # Updating fixed target weights every #target_update_frequency frames
                if self.training_metadata.frame % self.target_update_frequency == 0 and (self.training_metadata.frame != 0):
                    self.update_fixed_target_weights()

                if self.training_metadata.frame % freq == 0 and (self.training_metadata.frame != 0):
                    update = True
                # Choosing and performing action and updating the replay memory
                action = self.get_action(state, epsilon)
                next_state, reward, done, info = self.env.step(action)
                episode_frame += 1
                print("Frame {0} \t Action: {1} \t Reward: {2} \t State: {3}".format(self.training_metadata.frame, action, reward, next_state))

                # print("State: {0} \t Reward: {1}".format(next_state,reward))
                # if reward == 1:
                #     print(info['Fixed_Point'])
                #     utils.pause()

                # if isinstance(info['Fixed_Point'], np.ndarray):
                #     fp = info['Fixed_Point']
                #     self.writer.add_summary(self.sess.run(self.traj_summary,
                #         feed_dict={self.fp: [fp], self.P_x1:fp[0]}), self.training_metadata.frame)
                #     period_points = np.append(period_points,[fp],axis=0)

                # for i in range(np.size(next_state)):
                if self.state_shape[0]:
                    ns_in = next_state[-1][0]
                else:
                    ns_in = next_state[0]
                self.writer.add_summary(self.sess.run(self.scat_summary,
                    feed_dict={self.traj: ns_in}), self.training_metadata.frame)
                # traj = np.append(traj, [[state[0]]], axis=0)                    

                self.replay_memory.add(self, state, action, reward, next_state, done)

                # Performing experience replay if replay memory populated
                state = next_state
                done = info['true_done']
                self.sess.run(self.increment_frames_op)
                self.training_metadata.increment_frame()

                if self.replay_memory.length() > self.replay_memory.batch_size and self.training_metadata.frame %freq==0:#100 * self.replay_memory.batch_size:
                    self.experience_replay(alpha)

                # Creating q_grid if not yet defined and calculating average q-value
                if self.replay_memory.length() > 1000:
                    self.q_grid = self.replay_memory.get_q_grid(size=200, training_metadata=self.training_metadata)
                avg_q = self.estimate_avg_q()
                self.writer.add_summary(self.sess.run(self.training_summary, feed_dict={self.avg_q: avg_q}), self.training_metadata.frame)

            # end of episode
            # if self.replay_memory.length() > self.replay_memory.batch_size and update:
            #     print('Reach the end of an episode')
            #     self.experience_replay(alpha)
            #     utils.pause()

            if self.best_training_score==None or episode_frame<self.best_training_score:#score>self.best_training_score:
                self.best_training_score = episode_frame
                self.delete_previous_checkpoints()
                self.saver.save(self.sess, self.model_path + '/best.data.chkp', global_step=self.training_metadata.episode)
            if abs(self.training_metadata.num_episodes - episode)<10:
                self.saver.save(self.sess, self.model_path + '/last.data.chkp', global_step=self.training_metadata.episode)

            # record intermediate trajectories and fixed points
            temp_traj,x_bar = self.env.record_traj()
            np.savetxt(self.model_path+"/x_bar.csv", x_bar, delimiter=",")
            if self.training_metadata.episode%10 == 0:
                np.savetxt(self.model_path+"/temp_traj"+str(self.training_metadata.episode)+".csv", temp_traj, delimiter=",")

            # update tensorboard
            self.writer.add_summary(self.sess.run(self.test_summary,
                feed_dict={self.test_score: episode_frame}), self.training_metadata.episode)
            self.writer.add_summary(self.sess.run(self.frame_summary,
                feed_dict={self.f_count: self.training_metadata.frame}), self.training_metadata.episode)

            # training_scores = np.append(training_scores, [[self.training_metadata.episode, episode_frame]], axis=0)
            frame_eps = np.append(frame_eps, [[self.training_metadata.episode, self.training_metadata.frame]], axis=0)

        # end of training 
        # np.savetxt(self.model_path+"/training_scores.csv", training_scores, delimiter=",")
        np.savetxt(self.model_path+"/frame_eps.csv", frame_eps, delimiter=",")
        # np.savetxt(self.model_path+"/traj.csv", traj, delimiter=",")
        # np.savetxt(self.model_path+"/period_points.csv", period_points, delimiter=",")


    # Description: Tests the model
    # Parameters:
    # - num_test_episodes: 	Integer, giving the number of episodes to be tested over
    # - visualize: 			Boolean, gives whether should render the testing gameplay
    def test(self, num_test_episodes, visualize, pause=False):
        traj = np.empty((0,self.state_size),float)
        training_scores = np.empty((0,2),float)
        for episode in range(num_test_episodes):
            traj = np.append(traj, [[10,10]], axis=0)
            done = False
            state = self.env.reset(test=True)
            frame = 0
            # while not done:
            while frame < 1000:
                action = self.get_action(state, epsilon=0)
                next_state, reward, done, info = self.env.step(action, test=True)
                frame += 1
                traj = np.append(traj, [state], axis=0)
                state = next_state
                print("Reward: {0} \t State: {1} \t Fixed Point: {2}".format(reward, state, info['Fixed_Point']))
                done = info['true_done']
                if done: training_scores = np.append(training_scores, [[episode, frame]], axis=0)
                if pause: utils.pause()
        np.savetxt(self.model_path+"/test_training_scores.csv", training_scores, delimiter=",")
        np.savetxt(self.model_path+"/test_traj.csv", traj, delimiter=",")
        return traj, training_scores


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
