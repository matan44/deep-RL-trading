import numpy as np


class RandomEpisodeMarket:

    def __init__(self, sampler, window_state, open_cost, direction, risk_averse, additional_features):
        self.sampler = sampler
        self.window_state = window_state
        self.additional_features = additional_features
        self.open_cost = open_cost
        self.direction = direction
        self.risk_averse = risk_averse

        self.action_labels = ['empty', 'open', 'keep']
        self.n_action = len(self.action_labels)
        self.state_shape = (window_state + additional_features, self.sampler.n_var)
        self.t0 = window_state - 1
        self.empty = False
        self.t = None
        self._max_profit = 0
        self.price_df = None
        self.min_max_df = None
        self.title = None
        self.t_max = None

    @staticmethod
    def find_ideal(p, just_once):
        if not just_once:
            diff = p[1:] - p[:-1]
            return sum(np.minimum(np.zeros(diff.shape), diff))
        else:
            best = 0
            max_price = p[0]
            min_price = p[0]
            for price in p:
                if price < min_price:
                    min_price = price
                if price > max_price:
                    max_price = price
                    min_price = price
                delta = max_price - min_price
                if delta > best:
                    best = delta
            return best

    def get_ideal(self):
        return self.find_ideal(self.price_df.price.values[self.t:], just_once=True)

    def get_state(self):
        state = self.price_df.price.values[self.t - self.window_state:self.t]
        state_min = min(state)
        state = state - state_min
        t_timestamp = self.price_df.timestamp.values[self.t]
        min_max_state = self.min_max_df[
                            self.min_max_df.timestamp <= (t_timestamp - np.timedelta64(8, 'm'))
                            ].price.values[-self.additional_features:]
        if len(min_max_state) != self.additional_features:
            raise RuntimeError(
                'len(min_max_state) {} != {}. start_time_key = {}'.format(
                    len(min_max_state), self.additional_features, self.sampler.start_time_key
                )
            )
        min_max_state = min_max_state - state_min
        return np.concatenate([min_max_state, state])

    def get_valid_actions(self):
        if self.empty:
            return [0, 1]  # wait, open
        else:
            return [0, 2]  # close, keep

    def get_noncash_reward(self, empty=None):
        t = self.t
        if empty is None:
            empty = self.empty
        reward = self.direction * (self.price_df.price.values[t + 1] - self.price_df.price.values[t])
        if empty:
            reward -= self.open_cost
        if reward < 0:
            reward *= (1. + self.risk_averse)
        return reward

    def step(self, action):
        if action == 0:  # cash (wait/close)
            reward = 0.
            self.empty = True
        elif action == 1:  # open
            reward = self.get_noncash_reward()
            self.empty = False
        elif action == 2:  # keep
            reward = self.get_noncash_reward()
        else:
            raise ValueError('no such action: ' + str(action))
        self.t += 1
        return self.get_state(), reward, self.t == self.t_max, self.get_valid_actions()

    def reset(self, rand_price=True, start_time_key=None):
        self.empty = True
        self.t = self.t0 + 1
        if rand_price:
            self.price_df, self.min_max_df, start_time, self.title = self.sampler.sample(
                start_time_key=start_time_key)
            self.t_max = len(self.price_df) - 1
        self._max_profit = self.find_ideal(self.price_df.price.values[self.t:], just_once=True)
        return self.get_state(), self.get_valid_actions()
