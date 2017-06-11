import numpy as np

class LinearScheduler(object):
	def __init__(self, start_output, end_output, total_episodes):
		self.start_output = start_output
		self.end_output = end_output
		self.total_episodes = total_episodes

	def getParams(self):
		return {"start_output":self.start_output, 
		"end_output":self.end_output, 
		"total_episodes":self.total_episodes}

	def getScheduledValue(self, i):
		if i <= 0:
			return self.start_output
		elif i >= self.total_episodes:
			return self.end_output
		else:
			return (self.end_output-self.start_output)/self.total_episodes*i + \
			self.start_output

class ConstantScheduler(object):
	def __init__(self, output):
		self.output = output

	def getParams(self):
		return {"output":self.output}

	def getScheduledValue(self, i):
		return self.output

class InverseScheduler(object):
	def __init__(self, start_output):
		self.start_output = start_output

	def getParams(self):
		return {"start_output":self.start_output}

	def getScheduledValue(self, i):
		if i <= 0:
			return self.start_output
		else:
			return self.start_output / i

class ExponentialScheduler(object):
	def __init__(self, start_output, decay_episode):
		self.start_output = start_output
		self.decay_episode = decay_episode

	def getParams(self):
		return {"start_output":self.start_output, "decay_episode":self.decay_episode}

	def getScheduledValue(self, i):
		if i <= 0:
			return self.start_output
		else:
			return self.start_output * np.exp(-i / self.decay_episode)






