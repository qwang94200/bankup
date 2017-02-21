import numpy as py
import datetime
from budgea.plugins.forecasts.clusterer import TransactionClusterer
from budgea.plugins.forecasts.prediction import Prediction as PredictionCalculator
from budgea.models import Transaction, Category, Account
from budgea.tools.config import ConfigHandler
import logging

logging.basicConfig(level=logging.DEBUG)

import pandas as pd

config = ConfigHandler()
database = config.database

with database.session_scope(None) as session:
    main_cats = {}
    for cat in session.query(Category): ##got different cluster_id and category_id : cluster_id match to category_id
        main_cats[cat.id] = cat.id_parent_category
        
    acc = session.query(Account).get(723) ##account number ?
    records = acc.transactions.order_by(Transaction.rdate).all() ##list of transaction_ids
    ##for each record <transaction id=283280> : 1203
    ##category_label : bars/Sorties/Date: 2015-06-30/montant: -128.10/nature: cb/ wording: facture carte du 300515 hall beer brewe paris carte 49
    clusterer = TransactionClusterer(records, main_cats)
    clusterer.find_clusters(acc, datetime.date.today()) ##match
    ##

    prediction = PredictionCalculator()
    prediction.add_transactions(records, datetime.date.today().replace(day=1))
    prediction.compute_averages()
    for d in xrange(31):
        mean_amount, std_amount = prediction.get_prediction_still_to_be_spent(d+1)
        print '%02d: %s (%s)' % (d+1, mean_amount, std_amount)



