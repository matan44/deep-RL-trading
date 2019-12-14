import os
import time
import datetime
import numpy as np

from utils import makedirs


class Simulator:

	def play_episode(self, exploration, training=True, rand_price=True, print_t=False, get_ideal=False):
		print('play_episode')
		start_t = time.time()
		state, valid_actions = self.env.reset(rand_price=rand_price)
		done = False
		env_t = self.env.t + 1
		cum_rewards = [np.nan] * env_t
		actions = [np.nan] * env_t
		ideal = [np.nan] * env_t
		pqs = [[np.nan, np.nan, np.nan]] * env_t
		states = [None] * env_t
		prev_cum_rewards = 0.
		while not done:
			if print_t:
				print(self.env.t)

			action, pq = self.agent.act(state, exploration, valid_actions)
			next_state, reward, done, valid_actions = self.env.step(action)
			cum_rewards.append(prev_cum_rewards + reward)
			prev_cum_rewards = cum_rewards[-1]
			actions.append(action)
			states.append(next_state)
			pqs.append(pq)
			ideal.append(np.nan if not get_ideal else self.env.get_ideal())
			if training:
				self.agent.remember(state, action, reward, next_state, done, valid_actions)

			state = next_state

		if training:
			print('filled memory buffer', round(time.time() - start_t, 4), len(self.agent.memory))
		return cum_rewards, actions, states, pqs, ideal

	def train(self, n_episode, exploration_decay, exploration_min, print_t, exploration_init):
		fld_model = os.path.join(self.fld_save, 'model')
		makedirs(fld_model)	 # don't overwrite if already exists
		with open(os.path.join(fld_model,'QModel.txt'), 'w') as f:
			f.write(self.agent.model.qmodel)

		exploration = exploration_init
		fld_save = os.path.join(self.fld_save,'training')

		makedirs(fld_save)
		memory_filled = False
		first_replay = True
		n_training_episode = 0
		for n in range(n_episode):
			start_loop = time.time()
			print(f'\ntraining: {n + 1}...')
			exploration = max(exploration_min, exploration * exploration_decay)
			cum_rewards, actions, _, qs, ideal = self.play_episode(exploration, print_t=print_t)
			print('mem size, current mem', len(self.agent.memory),  self.agent.memory_size)
			if len(self.agent.memory) >= self.agent.memory_size:
				memory_filled = True

			if memory_filled:
				if n_training_episode % 64 == 0:
					print('starting replay ({}, {}): {}'.format(n, n_training_episode, datetime.datetime.now()))
					start_t = time.time()
					for i in range(self.replays):
						start_l = time.time()
						self.agent.replay()
						if first_replay:
							print('replayed one batch in', round(time.time() - start_l, 4))
					first_replay = False
					print('finished replay: {}'.format(datetime.datetime.now()))
					print('{} replays done. replay time: ------- {}'.format(self.replays, round(time.time() - start_t, 4)))
					start_l = time.time()
					self.agent.save(fld_model)
					print('saved model', round(time.time() - start_l, 4))
				n_training_episode += 1

			print('looped once in', round(time.time() - start_loop, 4))

	def test(self, n_episode, subfld):
		fld_save = os.path.join(self.fld_save, subfld)
		makedirs(fld_save)
		path_record = os.path.join(fld_save, 'record.csv')

		with open(path_record,'w') as f:
			f.write('episode,game,pnl,rel,MA\n')

		for n in range(n_episode):
			start_loop = time.time()
			print('testing...')
			cum_rewards, actions, states, pqs, ideal = self.play_episode(0, training=False, get_ideal=True)
			ss = [str(n), self.env.title.replace(',',';'),  str(cum_rewards[-1]), str(np.nanmedian(ideal))]
			with open(path_record, 'a') as f:
				f.write(','.join(ss)+'\n')
				print('\t'.join(ss))

			fig_path = os.path.join(fld_save, f'episode_{n}.png')
			self.visualizer.visualise_episode(self.env, cum_rewards, actions, pqs, ideal, fig_path)
			print('looped once in', round(time.time() - start_loop, 4))
		return path_record

	def __init__(self, agent, env, visualizer, fld_save, replays=24):
		self.replays = replays
		self.agent = agent
		self.env = env
		self.visualizer = visualizer
		self.fld_save = fld_save
