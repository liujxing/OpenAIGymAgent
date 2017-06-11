import numpy as np
import random

class TDLambdaTrainer(object):
	def __init__(self, env, lambda_value, exploration_scheduler, lr_scheduler):
		self.env = env
		self.lambda_value = lambda_value
		self._counter = 0
		self.q_value = np.zeros((self.env.observation_space.n, self.env.action_space.n))
		self.Z_value = np.zeros((self.env.observation_space.n, self.env.action_space.n))
		self.exploration_scheduler = exploration_scheduler
		self.lr_scheduler = lr_scheduler
		self._observation = self.env.reset()
		self._action = self.env.action_space.sample()


	def chooseAction(self, observation, counter):
		# choose an action based on epsilon-greedy algorithm
		p_exploration = self.exploration_scheduler.getScheduledValue(counter)
		if random.random() < p_exploration:
			action = self.env.action_space.sample()
		else:
			action = np.argmax(self.q_value[observation])
		return action

	def update(self, gamma=1.0):
		# calculate the learning rate
		lr = self.lr_scheduler.getScheduledValue(self._counter)
		# get next observation and next action
		next_observation, reward, done, info = self.env.step(self._action)
		next_action = self.chooseAction(next_observation, self._counter)
		# get difference in q-value
		q_delta = lr * (reward + gamma * self.q_value[next_observation, next_action] - \
			self.q_value[self._observation, self._action])
		# update eligibility trace of current q-state
		self.Z_value[self._observation, self._action] = 1
		# update q-value and eligibility trace of all q-state
		self.q_value += lr * q_delta * self.Z_value
		self.Z_value *= gamma * self.lambda_value
		# update current observation and current action
		if done:
			self._counter += 1
			self._observation = self.env.reset()
			self._action = self.chooseAction(self._observation, self._counter)
		else:
			self._observation = next_observation
			self._action = next_action



	def train(self, num_step=None, stop_threshold=None, gamma=1.0):
		if num_step is None and stop_threshold is None:
			raise ValueError("num_step and stop_threshold cannot be both None.")
		if num_step is None:
			num_step = np.inf
		if stop_threshold is None:
			stop_threshold = 0

		diff = np.inf
		i = 0
		while i<num_step:
			self.update(gamma=gamma)
			i+=1

	def retrivePolicy(self):
		return np.argmax(self.q_value, axis=1)

	def testPolicy(self, num_episode):
		optimum_actions = self.retrivePolicy()
		total_reward = 0
		for _ in range(num_episode):
			state = self.env.reset()
			done = False
			while not done:
				action = optimum_actions[state]
				state, reward, done, info = self.env.step(action)
				total_reward += reward
		return total_reward / num_episode














