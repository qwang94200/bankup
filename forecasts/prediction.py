# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np
from scipy.optimize import leastsq

from weboob.tools.log import getLogger


__all__ = ['Prediction']


class Prediction(object):
    def __init__(self, num_days=31, num_months=12, threshold_amount=100):
        assert num_days <= 31
        assert num_months <= 12

        self.num_days = num_days
        self.num_months = num_months
        self.threshold_amount = int(threshold_amount)

        self.month_ids = defaultdict(int)

        self.records_cum = 0
        self.records_mean = 0
        self.records_std = 0

        self.records_grouped = np.zeros((self.num_days, self.num_months))

        self.logger = getLogger('forecasts.prediction')

    def is_in_cluster(self, r):
        try:
            return r._isrec
        except AttributeError:
            return False

    def add_transactions(self, records, ignore_after=None):
        for r in records:
            day_id = r.rdate.day - 1
            month = r.rdate.month - 1

            # filter out positive amounts, large amounts and recurring
            if r.value > 0 or abs(r.value) > self.threshold_amount or self.is_in_cluster(r) or (ignore_after is not None and r.rdate >= ignore_after):
                continue

            if not (month in self.month_ids):
                assert len(self.month_ids) <= self.num_months
                self.month_ids[month] = len(self.month_ids)

            month_id = self.month_ids[month]

            self.records_grouped[day_id, month_id] += float(r.value)

    def compute_averages(self):
        records_rev = self.records_grouped[::-1, :]

        self.records_cum = np.cumsum(records_rev, axis=0)
        self.records_cum = self.records_cum[::-1, :]

        N = len(self.month_ids)

        if N > 1:
            self.records_mean = np.mean(self.records_cum[:, 0:N], axis=1).squeeze()
            self.records_std = np.std(self.records_cum[:, 0:N], axis=1).squeeze()
        else:
            self.records_mean = self.records_cum[:, 0]
            self.records_std = defaultdict(int)

    NB_DAYS = 5
    DIFF_STD = 1.5
    def adjust(self, real_data):
        pred_data = [(-(self.records_mean[0] - x)) for x in self.records_mean]

        count = 0
        nb_adjust = 0

        delta = real_data[-1] - pred_data[len(real_data)-1]

        for i in xrange(len(real_data)):
            if abs(real_data[i] - pred_data[i]) > (self.DIFF_STD * self.records_std[i]):
                count += 1
            else:
                count = 0

            if count > self.NB_DAYS:
                pred_params = self.get_plot_params(pred_data[:i+1])
                real_params = self.get_plot_params(real_data[:i+1])
                adjusted_data = [(-(self.records_mean[0] - x)) for x in self.records_mean]

                x = 0
                while x < len(self.records_mean):
                    adjusted_data[x] = adjusted_data[x] + (real_params[0] - pred_params[0]) * (x - len(real_data)-1) + delta
                    #self.records_mean[x] = self.records_mean[x] + (real_params[0] - pred_params[0]) * (x - len(real_data)-1) + delta #(real_params[1] - pred_params[1])
                    x += 1

                self.records_mean = [(-(adjusted_data[-1] - x)) for x in adjusted_data] # NOQA

                self.logger.debug('adjusted: %s %s' % (pred_params, real_params))
                nb_adjust += 1
                pred_data = [(-(self.records_mean[0] - x)) for x in self.records_mean]

                count = 0

        return nb_adjust


    def get_plot_params(self, data):
        x = np.arange(len(data))
        y = data

        if len(y) <= 1:
            return [0., 0.]

        fp = lambda v, x: v[0]*x + v[1]
        e = lambda v, x, y: (fp(v, x) - y)
        v0 = [30., 1.]

        v, success = leastsq(e, v0, args=(x, y))

        if success < 1 or success > 4:
            raise Exception('fail:(')

        return v

    def get_prediction_still_to_be_spent(self, day):
        assert day > 0
        assert day <= self.num_days

        day_id = day - 1

        return self.records_mean[day_id], self.records_std[day_id]
