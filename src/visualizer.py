import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


class Visualizer:

	def __init__(self, action_labels):
		self.n_action = len(action_labels)
		self.action_labels = action_labels

	def visualise_episode(self, env, cum_rewards, actions, pqs, ideal, fig_path):
		_, (ax_price, ax_action, ax_Q) = plt.subplots(3, 1, sharex='all', figsize=(14, 14))
		p = env.price_df.price.values - env.price_df.price.values[-1]
		ax_price.plot(p, 'k-', label='prices')
		ax_price.plot(cum_rewards, 'b', label='P&L')
		ax_price.plot(ideal, 'r', label='ideal P&L')
		ax_price.legend(loc='best', frameon=False)
		ax_price.set_title(env.title + f', explored: {cum_rewards[-1]}, median ideal: {np.nanmedian(ideal)}')

		ax_action.set_title('Actions: cash=0, open=1, close=2')
		ax_action.plot(actions, 'b', label='explored')
		ax_action.set_ylim(-0.4, self.n_action - 0.6)
		ax_action.set_ylabel('action')
		ax_action.set_yticks(range(self.n_action))
		ax_action.legend(loc='best', frameon=False)
		
		styles = ['k', 'r', 'b']
		qs_df = pd.DataFrame(pqs, columns=['cash', 'open', 'keep'])
		for column, style in zip(qs_df.columns, styles):
			ax_Q.plot(qs_df[column], style, label=column)

		ax_Q.set_ylabel('Q')
		ax_Q.legend(loc='best', frameon=False)
		ax_Q.set_xlabel('t')

		plt.subplots_adjust(wspace=0.4)
		print('fig_path', fig_path)
		plt.savefig(fig_path)
		plt.close()
