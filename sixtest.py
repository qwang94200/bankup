import numpy as py
import pandas as pd
import datetime
from budgea.plugins.forecasts.clusterer import TransactionClusterer
from budgea.plugins.forecasts.prediction import Prediction as PredictionCalculator
from budgea.models import Transaction, Category, Account
from budgea.tools.config import ConfigHandler
import logging
LOG_FILENAME = 'example.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

config = ConfigHandler()
database = config.database

##date_threshold=datetime.date(2017,1,1)
date_threshold=datetime.date.today().replace(day=1)
date_threshold=datetime.date(2016,12,1)

data = pd.read_csv('sqlquery/select_profile.csv', delimiter=',')
selected_id_account=data['id_account'].tolist()

import pickle

with database.session_scope(None) as session:
    main_cats = {}
    for cat in session.query(Category): ##got different cluster_id and category_id : cluster_id match to category_id
        main_cats[cat.id] = cat.id_parent_category
        
    for id in selected_id_account:
        logging.debug('###Starting id account: %d' %id)
        totrecords=pickle.load(open(str(id), "rb"))
        records=[]
        for r in totrecords:
            if r.rdate<datetime.date(2016,12,1) and r.rdate>=datetime.date(2016,6,1):
                records.append(r)
        acc=None
        clusterer = TransactionClusterer(records, main_cats, date_threshold) ##clustering
        clusterer.find_clusters(acc, date_threshold)
        
