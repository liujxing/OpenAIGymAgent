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

