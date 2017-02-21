# -*- coding: utf-8 -*-

from decimal import Decimal
from datetime import timedelta

import numpy as np
import scipy as sp

from sklearn.cluster import DBSCAN

from weboob.tools.log import getLogger

class Transaction(object):
    def __init__(self, simplified_wording, rdate, value, id_category):
        self.id = None
        self.id_cluster = None
        self.simplified_wording = simplified_wording
        self.rdate = rdate
        self.value = value
        self.id_category = id_category

class Cluster(object):
    def __init__(self):
        self.id = None
        self.records = []
        self.core_records = {}
        self.mean_increment = None
        self.median_increment = None
        self.std_increment = None
        self.mean_day = None
        self.std_day = None
        self.mean_amount = None
        self.std_amount = None

        self.wording = None
        self.category_id = None
        self.next_date = None

        self.logger = getLogger('forecasts.cluster')

    def add_records(self, records):
        count = 0
        self.records = [tr for tr in self.records if tr.id is not None]
        already_ids = set([tr.id for tr in self.records])
        for tr in records:
            if not tr.id in already_ids:
                self.add_record(tr)
                count += 1

        self.refresh()
        self.find_holes()

        return count

    def add_record(self, r):
        if r.id in [tr.id for tr in self.records]:
            return

        r._isrec = True
        self.records.append(r)
        self.records.sort(key=lambda x: x.rdate, reverse=True)

    def add_core_record(self, r):
        if self.id is None:
            self.id = r.id
        self.core_records[r.id] = r

    FLAG_INCREMENT = 0x01
    FLAG_DAY       = 0x02
    FLAG_AMOUNT    = 0x04
    FLAG_CATEGORY  = 0x08
    FLAG_WORDING   = 0x10
    FLAG_NEXT_DATE = 0x20
    FLAG_ALL       = 0xff
    def refresh(self, flags=FLAG_ALL):
        dates = sorted([r.rdate for r in self.records])

        if flags & self.FLAG_INCREMENT and len(self.records) >= 2:
            increments = np.array([(dates[i] - dates[j]).days \
                                    for i,j in zip(range(1, len(dates)), range(0, len(dates)-1))])

            self.mean_increment = np.mean(increments)
            self.median_increment = np.median(increments)
            self.std_increment = np.std(increments)

        if flags & self.FLAG_DAY:
            days = np.array([d.day for d in dates])
            self.mean_day = np.mean(days)
            self.std_day = np.std(days)

        if flags & self.FLAG_AMOUNT:
            amounts = np.array([float(r.value) for r in self.records])
            self.mean_amount = np.mean(amounts)
            self.std_amount = np.std(amounts)

        if flags & self.FLAG_CATEGORY:
            for record in self.records:
                if record.id_category is not None:
                    self.category_id = record.id_category

        if flags & self.FLAG_WORDING:
            self.wording = self.records[0].simplified_wording
        if flags & self.FLAG_NEXT_DATE and self.median_increment is not None:
            self.next_date = self.records[0].rdate + timedelta(days=self.median_increment)

    def find_holes(self):
        self.logger.debug('\n--> CLUSTER %s' % self.wording.encode('utf-8', 'replace'))
        self.logger.debug('%s %s %s' % (self.mean_increment, self.median_increment, self.std_increment))

        r1 = 0
        r2 = 1
        t1 = None
        t2 = None
        while r2 < len(self.records):
            t1 = self.records[r1]
            t2 = self.records[r2]
            d = (t1.rdate - t2.rdate).days
            self.logger.debug('%s %s %s' % (t1.value, t1.rdate, t1.simplified_wording.encode('utf-8', 'replace')))
            self.logger.debug('distance: %s' % d)
            if r1 > 0 and d > 1 and round(d/self.median_increment) == 2.0:
                self.logger.debug('missing transaction (%s in %s)' % (d, self.median_increment))
                self.records.insert(r2, Transaction(simplified_wording='ADDED %s' % t1.simplified_wording,
                                                    rdate=t1.rdate - timedelta(days=d/2),
                                                    value=self.mean_amount,
                                                    id_category=t1.id_category))
                continue

            r1 += 1
            r2 += 1
        if t2 is not None:
            self.logger.debug('%s %s %s' % (t2.value, t2.rdate, t2.simplified_wording.encode('utf-8', 'replace')))
        self.refresh()

    def find_missing_transaction(self, acc, today):
        return
        next_date = self.next_date
        while self.median_increment is None or next_date <= (today + timedelta(days=self.median_increment)):
            delta_date = self.median_increment/5 if self.median_increment is not None else 3
            delta_value = self.std_amount if self.std_amount is not None else self.mean_amount/20

            params = {'date':           next_date,
                      'begin_date':     next_date - timedelta(days=delta_date),
                      'end_date':       next_date + timedelta(days=delta_date),
                      'value':          self.mean_amount,
                      'begin_value':    round(Decimal('%.2f' % self.mean_amount) - Decimal('%.2f' % abs(delta_value))),
                      'end_value':      round(Decimal('%.2f' % self.mean_amount) + Decimal('%.2f' % abs(delta_value))) + 1,
                      'wording':        self.wording.upper(),
                      'stemmed':        (self.records[0].stemmed_wording if len(self.records) > 0 else self.wording).upper(),
                      'category_id':    self.category_id,
                     }

            conds = [
                     '(stemmed_wording = :stemmed OR simplified_wording = :wording) AND value = :value AND date(rdate) >= :begin_date AND date(rdate) <= :end_date',
                     '(stemmed_wording = :stemmed OR simplified_wording = :wording) AND value >= :begin_value AND value <= :end_value AND date(rdate) = :date',
                     '(stemmed_wording = :stemmed OR simplified_wording = :wording) AND value >= :begin_value AND value <= :end_value AND date(rdate) >= :begin_date AND date(rdate) <= :end_date',
                     'value = :value AND date(rdate) >= :begin_date AND date(rdate) <= :end_date',
                     'value >= :begin_value AND value <= :end_value AND date(rdate) = :date',
                     '(stemmed_wording = :stemmed OR simplified_wording = :wording) AND date(rdate) >= :begin_date AND date(rdate) <= :end_date',
                     'value >= :begin_value AND value <= :end_value AND date(rdate) >= :begin_date AND date(rdate) <= :end_date AND id_category = :category_id',
                     '(LOCATE(:stemmed, stemmed_wording) OR LOCATE(:wording, simplified_wording)) AND date(rdate) >= :begin_date AND date(rdate) <= :end_date AND id_category = :category_id',
                    ]
            for cond in conds:
                tr = acc.transactions.filter(cond).params(params).first()
                if tr is not None:
                    self.logger.debug('Add transaction %s' % tr.simplified_wording.encode('utf-8', 'replace'))
                    self.add_record(tr)
                    return True

            if self.median_increment is None or self.median_increment < 1:
                break
            next_date += timedelta(days=self.median_increment)
        return False

class TransactionClusterer(object):
    """Clustering of transactions

    """

    RECORD_TYPES = ['defaut','inconnu','prelevement','cheque','depot','remboursement','virement','retrait','cb']

    def __init__(self, records, main_cats, eps = 0.1, min_elements = 3):
        """Initialize clustering.

        records -- list of user transactions
        eps -- neighborhood radius
        min_elements -- minimum elements to form a cluster

        """

        self.records = records
        self.main_cats = main_cats
        self.min_elements = min_elements
        self.eps = eps
        self.clusters = []
        self.logger = getLogger('forecasts.clusterer')

    def find_cluster(self, ids):
        for c in self.clusters:
            for r in c.core_records.iterkeys():
                if r in ids:
                    return c

    def get_features(self, condition=0):
        """Compute features for each transaction and return a matrix of size #transactions x #features.

        """

        num_types = len(self.RECORD_TYPES)

        X = np.zeros((len(self.records), 2 + num_types + max(self.main_cats.values())))

        if condition & self.COND_VALUE:
            X[:, 0] = [float(rec.value)/1.0 for rec in self.records]
        else:
            X[:, 0] = [10.0 if rec.value > 0 else -10.0 for rec in self.records]
        #X[:, 1] = [100 * rec.rdate.day/31.0]
        #X[:, 1] = [rec.rdate.day/620.0  for rec in self.records]

        i = 0

        value_type = 1
        value_cat = 10

        for r in self.records:
            try:
                type = self.RECORD_TYPES.index(r.nature) + 1
            except ValueError:
                type = 0

            X[i, 2 + type - 1] = value_type

            if r.id_category == 0:
                X[i, (2 + num_types):] = value_cat
            else:
                X[i, (2 + num_types) + self.main_cats.get(r.id_category, 0) - 1] = value_cat

            i = i + 1

        if condition & self.COND_LABEL:
            X = np.hstack([X, self.get_label_features()])

        return X

    def get_label_features(self, threshold_frequent = 1.0, threshold_infrequent = 0.9):
        """ Return a list of words to be used in label features.

        This scans through all the available labels, extract tokens, filter and return a list of words.

        threshold_frequent -- quantile over which to filter out words (that are too frequent, such as, e.g., stopwords)
        threshold_infrequent -- quantile under which to filter out words (that are too infrequent)

        """

        # go through all tokens, filter a little
        max_voc = 4 * len(self.records)

        bags = np.zeros((len(self.records), max_voc))
        num_words = 0
        i = 0
        vocabulary = {}

        for r in self.records:
            words = r.simplified_wording.split()
            for word in words:
                if num_words >= max_voc:
                    break

                lw = word.lower()

                if lw.isalpha() and len(word) >= 3:
                    if not (lw in [w for w in vocabulary.iterkeys()]):
                        vocabulary[lw] = num_words
                        num_words += 1

                    bags[i, vocabulary[lw]] = 1

            i += 1

        counts = np.sum(bags > 0, axis = 0)

        # filter out too (in)frequent words
        count_frequent = sp.stats.scoreatpercentile(counts, 100 * threshold_frequent)
        count_infrequent = sp.stats.scoreatpercentile(counts, 100 * threshold_infrequent)

        bags_filtered = bags[:, (counts > count_infrequent) & (counts < count_frequent)]

        return bags_filtered

    def get_clusters(self, condition=0):
        """Perform clustering and return a (representatives, cluster ids) tuple.

        """

        if len(self.records) == 0:
            return (), ()

        db = DBSCAN(eps=self.eps, min_samples=self.min_elements).fit(self.get_features(condition))
        core_samples = db.core_sample_indices_
        labels = db.labels_

        return (core_samples, labels)

    def mark_records(self, condition=0):
        """Go through all transactions and mark as recurring those that were put in a cluster.

        This uses helper function mark_cluster, and also fills out some extra information.
        """

        self.logger.debug('====== MARK RECORDS =======')

        all_records = self.records
        records = []
        for r in self.records:
            if (not (condition & self.COND_POSITIVE) or r.value > 0) and \
               (not (condition & self.COND_BIG) or abs(r.value) >= 100):
                records.append(r)
        self.records = records

        (core_samples, labels) = self.get_clusters(condition)

        to_remove = set()
        for label in set(labels):
            if label == -1:
                continue

            class_members = [index[0] for index in np.argwhere(labels == label)]
            cluster_core_samples = [index for index in core_samples \
                                    if labels[index] == label]

            cluster = Cluster()

            for r in class_members:
                cluster.add_record(self.records[r])
                to_remove.add(self.records[r].id)

            for r in cluster_core_samples:
                cluster.add_core_record(self.records[r])

            cluster.refresh()
            cluster.find_holes()
            self.clusters.append(cluster)

        records = []
        for r in all_records:
            if not r.id in to_remove:
                records.append(r)
        self.records = records

    COND_POSITIVE = 0x01
    COND_LABEL    = 0x02
    COND_VALUE    = 0x04
    COND_BIG      = 0x08
    def find_clusters(self, acc, today):
        passes = [self.COND_LABEL|self.COND_VALUE,
                  self.COND_BIG|self.COND_LABEL,
                  self.COND_BIG|self.COND_VALUE]

        for cond in passes:
            self.mark_records(cond)
            self.post_process(acc, today)

    def post_process(self, acc, today):
        clusters = []
        for cluster in self.clusters:
            if (cluster.records[0].rdate < (today - timedelta(days=(cluster.median_increment * 2 + cluster.std_increment)))) or cluster.median_increment < 1 or \
               not ((28 <= cluster.median_increment <= 33 and cluster.std_increment < (cluster.median_increment/2.0)) or cluster.std_increment < (cluster.median_increment * 0.12)):
                self.logger.debug('### Remove cluster for %s' % cluster.wording.encode('utf-8', 'replace'))
                for tr in cluster.records:
                    tr._isrec = False
                    if tr.id is not None:
                        self.records.append(tr)
            else:
                self.logger.debug('### Add cluster for %s' % cluster.wording.encode('utf-8', 'replace'))
                clusters.append(cluster)
                if (cluster.next_date + timedelta(days=(1*cluster.median_increment/4))) < today:
                    cluster.find_missing_transaction(acc, today)
                    cluster.refresh()

            self.logger.debug('%s %s' % (cluster.median_increment, cluster.std_increment))
            for r in cluster.records:
                self.logger.debug('%s %s %s' % (r.value, r.rdate, r.simplified_wording.encode('utf-8', 'replace')))

        self.clusters = clusters

    def add_old_cluster(self, c, records, acc, today):
        cluster = Cluster()

        for r in records:
            cluster.add_record(r)
            cluster.add_core_record(r)

        if len(records) >= 1:
            cluster.refresh()
            cluster.find_holes()

        cluster.next_date = c.next_date
        cluster.wording = c.wording
        cluster.category_id = c.id_category
        cluster.median_increment = c.median_increment
        cluster.mean_amount = c.mean_amount

        if cluster.mean_increment is not None or len(records) == 0:
            if cluster.find_missing_transaction(acc, today):
                flags = cluster.FLAG_ALL
                if cluster.mean_increment is None:
                    flags &= ~cluster.FLAG_WORDING
                cluster.refresh()

        if (cluster.next_date + timedelta(days=60)) < today:
            # if this is really in future, disable the cluster
            return None

        self.logger.debug('added!')
        for r in cluster.records:
            self.logger.debug('%s %s %s' % (r.value, r.rdate, r.simplified_wording.encode('utf-8', 'replace')))

        self.clusters.append(cluster)
        return cluster
