from collections import deque

import random, os, pickle
import numpy as np

from utils import makedirs


class Agent:

	def __init__(self, model, batch_size=32, discount_factor=0.95):

		self.model = model
		self.batch_size = batch_size
		self.discount_factor = discount_factor
		self.memory_size = 100 * 256
		self.memory = deque(maxlen=self.memory_size)

	def remember(self, state, action, reward, next_state, done, next_valid_actions):
		self.memory.append((state, action, reward, next_state, done, next_valid_actions))

	def replay(self):
		batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
		for state, action, reward, next_state, done, next_valid_actions in batch:
			qv = reward
			if not done:
				predicted_valid_actions, _ = self.get_q_valid(next_state, next_valid_actions)
				qv += self.discount_factor * np.nanmax(predicted_valid_actions)
			self.model.fit(state, action, qv)

	def get_q_valid(self, state, valid_actions):
		pq = self.model.predict(state)
		if len(pq) != 3:
			raise RuntimeError('Unexpected model output shape')
		q_valid = [np.nan] * 3
		for action in valid_actions:
			q_valid[action] = pq[action]
		return q_valid, pq

	def act(self, state, exploration, valid_actions):
		if np.random.random() > exploration:
			q_valid, pq = self.get_q_valid(state, valid_actions)
			if np.nanmin(q_valid) != np.nanmax(q_valid):
				return np.nanargmax(q_valid), pq
		return random.sample(valid_actions, 1)[0], [np.nan] * 3

	def save(self, fld):
		makedirs(fld)
		attr = {'batch_size': self.batch_size, 'discount_factor': self.discount_factor}
		pickle.dump(attr, open(os.path.join(fld, 'agent_attr.pickle'),'wb'))
		self.model.save(fld)

	def load(self, fld):
		path = os.path.join(fld, 'agent_attr.pickle')
		print(path)
		attr = pickle.load(open(path,'rb'))
		for k in attr:
			setattr(self, k, attr[k])
		self.model.load(fld)
