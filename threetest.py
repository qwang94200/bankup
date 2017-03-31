import numpy as py
import pandas as pd
import datetime
from budgea.plugins.forecasts.clusterer import TransactionClusterer
from budgea.plugins.forecasts.prediction import Prediction as PredictionCalculator
from budgea.models import Transaction, Category, Account
from budgea.tools.config import ConfigHandler
from data_extract import extract_compte_cheque

import logging
LOG_FILENAME = 'example.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

config = ConfigHandler()
database = config.database

##date_threshold=datetime.date(2017,1,1)
date_threshold=datetime.date.today().replace(day=1)
date_threshold=datetime.date(2016,12,1)

number_count, count_names = extract_compte_cheque.get_datas()

"""
with database.session_scope(None) as session:
    main_cats = {}
    for cat in session.query(Category): ##got different cluster_id and category_id : cluster_id match to category_id
        main_cats[cat.id] = cat.id_parent_category
        
    for i, id in enumerate(count_names):
        logging.debug('###Starting id account: %d' %id)
        totrecords = extract_compte_cheque.get_data_compte_cheque(i)
        
        records=[]
        for r in totrecords:
            if r.rdate<datetime.date(2016,12,1) and r.rdate>=datetime.date(2016,9,1):
                records.append(r)
        acc=None
        clusterer = TransactionClusterer(records, main_cats, date_threshold) ##clustering
        clusterer.find_clusters(acc, date_threshold)
        

"""
with database.session_scope(None) as session:
    main_cats = {}
    for cat in session.query(Category): ##got different cluster_id and category_id : cluster_id match to category_id
        main_cats[cat.id] = cat.id_parent_category

    for id in count_names:
        acc = session.query(Account).get(int(id)) ##account number ?
        all_records = acc.transactions.order_by(Transaction.rdate).all()
        records=[r for r in all_records if r.rdate>=date_threshold and r.rdate<=date_threshold.replace(month=1)]
        
        acc=None
        clusterer = TransactionClusterer(records, main_cats) ##clustering
        clusterer.find_clusters(acc, date_threshold)
        
        prediction = PredictionCalculator()
        prediction.add_transactions(records, datetime.date.today().replace(day=1))
        prediction.compute_averages()
        y_predict=[]
        for d in xrange(31):
            mean_amount, std_amount = prediction.get_prediction_still_to_be_spent(d+1)
            #print '%02d: %s (%s)' % (d+1, mean_amount, std_amount)
            y_predict.append([d+1, mean_amount, std_amount])
        y_predict=py.array(y_predict).reshape(len(y_predict), 3)
        
        # y_real=[]
        # for r in records:
        #     y_real.append([r.rdate.day, r.rdate.month, r.rdate.year, r.value])
        # y_real=py.array(y_real).reshape(len(y_real), 4)
