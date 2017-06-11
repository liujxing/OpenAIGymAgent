from trainer import TDLambdaTrainer
from scheduler import *
import gym
from gym import wrappers
import argparse
import time

def main(env, record=False, upload=False):
	env = gym.make(env)
	env.reset()

	lr_scheduler = ConstantScheduler(0.05)
	#exploration_scheduler = LinearScheduler(1.0, 0.2, 50000)
	exploration_scheduler = ExponentialScheduler(1.0, 6000000)
	q_trainer = TDLambdaTrainer(env=env, lambda_value=0.75, 
		exploration_scheduler=exploration_scheduler, 
		lr_scheduler=lr_scheduler)

	for i in range(200):
		start = time.time()
		q_trainer.train(num_step=200000, gamma=0.99)
		print("iteration %d average reward %.2f time %d seconds" \
			%(i, q_trainer.testPolicy(2000), time.time()-start))

	
	if record or upload:
		q_trainer.env = wrappers.Monitor(env, '/tmp/FrozenLake-experiment-1', force=True)
		print("Average reward per episode: %.2f" %(q_trainer.testPolicy(5000)))
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
