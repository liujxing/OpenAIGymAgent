from trainer import QTrainer
from scheduler import *
import gym
from gym import wrappers
import argparse
import time

def main(env, record=False, upload=False):
	env = gym.make(env)
	env.reset()

	#lr_scheduler = LinearScheduler(3e-1, 1e-1, 2000)
	lr_scheduler = ConstantScheduler(0.1)
	exploration_scheduler = LinearScheduler(1.0, 0.2, 200000)
	q_trainer = QTrainer(env, exploration_scheduler, lr_scheduler)

	for i in range(200):
		start = time.time()
		q_trainer.train(num_step=60000)
		print("episode %d average reward %.2f" %(i, q_trainer.testPolicy(2000)))
		print("time %d seconds" %(time.time()-start))


	if record or upload:
		q_trainer.env = wrappers.Monitor(env, '/tmp/FrozenLake-experiment-1', force=True)
		print("Average reward per episode: %.2f" %(q_trainer.testPolicy(4000)))
	if upload:
		env.close()
		q_trainer.env.close()
		gym.upload("/tmp/FrozenLake-experiment-1")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("env", help="OpenAI gym environment", type=str)
	parser.add_argument("--record", help="whether to record the testing process", default=False, type=bool)
	parser.add_argument("--upload", help="whether to upload the testing process", default=False, type=bool)
	args = parser.parse_args()		
	main(env=args.env, record=args.record, upload=args.upload)
