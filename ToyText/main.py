from trainer import QTrainer
from scheduler import LinearScheduler
import gym
from gym import wrappers

def main(upload=False, api_key=None):
	env = gym.make("FrozenLake-v0")
	env = wrappers.Monitor(env, '/tmp/FrozenLake-experiment-1', force=True)
	env.reset()

	lr_scheduler = LinearScheduler(1e-1, 1e-1, 500)
	exploration_scheduler = LinearScheduler(1.0, 0.1, 1000)
	q_trainer = QTrainer(env, exploration_scheduler, lr_scheduler)
	q_trainer.train(num_step=4000)
	print("Optimum policy: %s" %(q_trainer.retrivePolicy()))
	print("Q-value: %s" %(q_trainer.q_value))
	env.close()
	if upload and api_key is not None:
		gym.upload("/tmp/FrozenLake-experiment-1", api_key=api_key)

if __name__ == "__main__":
	#main(upload=True, api_key="sk_Ah0oo5XcTEWHFwckzi7kUA")
	main()
