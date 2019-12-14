import random, os

import numpy as np
import pandas as pd


class RandomEpisodeSampler:

    def __init__(self, train_dates, test_dates, source_path, mode='train'):
        self.title = 'random_episode_sampler_eurusd'
        self.n_var = 1
        self.high_res_window = 80
        self.low_res_window = 16
        self.window_episode = self.high_res_window + self.low_res_window
        self.train_dates = train_dates
        self.test_dates = test_dates
        self._start_time = None

        if mode == 'train':
            dates = self.train_dates
        elif mode == 'out-of-sample':
            dates = self.test_dates
        else:
            raise ValueError('Unknown mode {}'.format(mode))
        self._segments = pd.concat([pd.read_feather(os.path.join(
            source_path, '/episodes_v0/{}_{}_{}'.format('eur_usd', date.strftime('%Y%m%d'), '6s_segments')
        )) for date in dates]).reset_index(drop=True)
        self._min_max = pd.concat([
            pd.read_feather(os.path.join(
                source_path, '/episodes_v0', '{}_{}_{}'.format('eur_usd', date.strftime('%Y%m%d'), '300s_min_max')
            ))[['point_1_timestamp', 'point_1_price']]
            for date in dates
        ]).reset_index(drop=True).rename({'point_1_timestamp': 'timestamp', 'point_1_price': 'price'}, axis=1)
        self.start_times = self._min_max[
            (
                (self._min_max.timestamp.dt.hour > 7) |
                (
                    (self._min_max.timestamp.dt.minute > 30) &
                    (self._min_max.timestamp.dt.hour > 6)
                )
            ) &
            (self._min_max.timestamp.dt.hour < 19)
        ][['timestamp']].copy().reset_index(drop=True)
        self.shuffled_start_time_keys = list(range(len(self.start_times)))
        random.shuffle(self.shuffled_start_time_keys)
        self.start_time_key = None

    def get_episode_data(self, start_time):
        episode_segments = self._segments[
            (self._segments.timestamp <= (start_time + np.timedelta64(12, 'm'))) &
            (self._segments.timestamp >= (start_time - np.timedelta64(8, 'm')))
            ].reset_index(drop=True)
        episode_min_max = self._min_max[
            (self._min_max.timestamp <= (start_time + np.timedelta64(12, 'm'))) &
            (self._min_max.timestamp >= (start_time - np.timedelta64(60, 'm')))
            ].reset_index(drop=True)
        return episode_segments, episode_min_max

    def sample(self, start_time_key=None):
        if start_time_key is None:
            if not self.shuffled_start_time_keys:
                self.shuffled_start_time_keys = list(range(len(self.start_times)))
                random.shuffle(self.shuffled_start_time_keys)
            self.start_time_key = self.shuffled_start_time_keys.pop()
        else:
            self.start_time_key = start_time_key
        self._start_time = self.start_times.loc[self.start_time_key].timestamp
        segments, min_max = self.get_episode_data(self._start_time)
        return segments, min_max, self._start_time, str(pd.to_datetime(self._start_time))
