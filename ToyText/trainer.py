import numpy as np
import random
from scheduler import LinearScheduler

class QTrainer(object):
	def __init__(self, env, exploration_scheduler, lr_scheduler):
		self.env = env
		self._counter = 0
		#self.q_value = np.random.uniform(size=(self.env.observation_space.n, self.env.action_space.n))
		#self.q_value_old = np.random.uniform(size=(self.env.observation_space.n, self.env.action_space.n))
		self.q_value = np.zeros((self.env.observation_space.n, self.env.action_space.n))
		self.q_value_old = np.zeros((self.env.observation_space.n, self.env.action_space.n))
		self.exploration_scheduler = exploration_scheduler
		self.lr_scheduler = lr_scheduler
		self._observation = self.env.reset()

	def envStep(self, counter):
		# choose an action based on epsilon-greedy algorithm
		p_exploration = self.exploration_scheduler.getScheduledValue(counter)
		if random.random() < p_exploration:
			action = self.env.action_space.sample()
		else:
			action = np.argmax(self.q_value[self._observation])
		observation, reward, done, info = self.env.step(action)
		return action, observation, reward, done

	def update(self, counter, prev_observation, next_observation, action, reward, gamma=1.0):
		# get the learning rate 
		lr = self.lr_scheduler.getScheduledValue(counter)
		self.q_value[prev_observation, action] += lr * (reward + \
			gamma * max(self.q_value[next_observation]) - \
			self.q_value[prev_observation, action])


	def calculateQChange(self):
		return np.max(np.abs(self.q_value - self.q_value_old))

	def trainStep(self):
		action, next_observation, reward, done = self.envStep(self._counter)
		self.update(self._counter, self._observation, next_observation, action, reward)
		self._observation = next_observation
		# reset the environment
		if done:
			self._observation = self.env.reset()
			self._counter += 1


	def train(self, num_step=None, stop_threshold=None):
		if num_step is None and stop_threshold is None:
			raise ValueError("num_step and stop_threshold cannot be both None.")
		if num_step is None:
			num_step = np.inf
		if stop_threshold is None:
			stop_threshold = 0

		diff = np.inf
		i = 0
		while i<num_step:
		#while i<num_step and diff>stop_threshold:
			self.q_value_old[:] = self.q_value[:]
			self.trainStep()
			diff = self.calculateQChange()
			i+=1
			print(i, diff)

	def retrivePolicy(self):
		return np.argmax(self.q_value, axis=1)










