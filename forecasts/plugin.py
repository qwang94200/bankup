# -*- coding: utf-8 -*-

from itertools import chain
from dateutil.parser import parse as parse_date
from decimal import Decimal
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta

from sqlalchemy.orm import object_session
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.sql import func

from werkzeug.routing import Map, Rule

from budgea.tools.datetime import next_month, first_month_day, last_month_day
from budgea.tools.report import UserAlert
from budgea.api.response import JsonResponse
from budgea.api.rest import RESTResource, AccountsRequired
from budgea.models import Account, TransactionsCluster, Prediction, Transaction, Category
from budgea.tools.plugin import Plugin
from budgea.plugins.user import UserResource, MeUserResource

from .clusterer import TransactionClusterer
from .prediction import Prediction as PredictionCalculator


__all__ = ['ForecastsPlugin']


class ForecastResource(AccountsRequired, RESTResource):
    url_map = Map([Rule('/', endpoint='index')], strict_slashes=False)

    parents = {
        UserResource:       '/<string:id_user>/forecast',
        MeUserResource:     '/forecast',
    }

    def projection_categories(self, query, categories, start_date, end_date, income):
        s = Decimal('0')

        for cluster in query(TransactionsCluster):
            if not cluster.enabled or (income is not None and ((cluster.mean_amount < 0) == income)):
                continue

            transactions = query(Transaction).filter(Transaction.id_cluster == cluster.id, Transaction.rdate.between(start_date, end_date))

            tables = []
            for tr in transactions:
                table = {'text': tr.wording,
                         'tdate': tr.rdate,
                         'debited': True,
                         'disabled': False,
                         'thisMonth': True,
                         'category': cluster.category,
                         'value': tr.value,
                         'cluster': cluster,
                        }
                tables.append(table)

            current_date = cluster.next_date
            interval = None
            if cluster.median_increment:
                interval = timedelta(days=cluster.median_increment)

            today = datetime.date.today()
            while (interval and current_date <= end_date) or len(tables) == 0:
                disabled = False
                if current_date and current_date < today:
                    if (today - current_date).days > (cluster.median_increment/2 if cluster.median_increment else 3):
                        disabled = True
                    else:
                        current_date = today + timedelta(days=1)

                table = {'text':        cluster.wording,
                         'tdate':       current_date if cluster.next_date else None,
                         'debited':     False,
                         'thisMonth':   current_date and current_date <= end_date,
                         'disabled':    disabled,
                         'category':    cluster.category,
                         'value':       cluster.mean_amount,
                         'cluster':     cluster,
                        }
                tables.append(table)

                if not disabled and table['thisMonth']:
                    s += int(round(cluster.mean_amount))

                if interval:
                    current_date += interval
                else:
                    break

            for table in tables:
                if cluster.id_category in categories:
                    categories[cluster.id_category].append(table)
                else:
                    categories[cluster.id_category] = [table]

        return s


    def compute_projection(self, query, start_date, end_date, income = False):
        projection = {'categories': {},
                     }
        sumA = self.projection_categories(query, projection['categories'], start_date, end_date, income)
        projection['catsum'] = int(round(abs(sumA)))

        predSumL = predSumH = predSumA = Decimal('0')
        for p in query(Prediction).filter(Prediction.day == datetime.date.today().day):
            if income:
                continue

            predSumL += p.mean_amount - p.std_amount
            predSumH += p.mean_amount + p.std_amount
            predSumA += p.mean_amount
            sumA += p.mean_amount

        projection['predsumL'] = int(round(abs(predSumL)))
        projection['predsumH'] = int(round(abs(predSumH)))
        projection['predsumA'] = int(round(abs(predSumA)))
        projection['sum'] = int(round(abs(sumA)))

        return projection

    def get_collection(self, request, query, **values):
        query.add_filter((Account.display == True) & (Account.disabled == None))
        query.add_join(Account)

        today = datetime.date.today()
        start_date = first_month_day(today)
        end_date = last_month_day(start_date)

        tq = query(func.sum(Transaction.value)).filter(Transaction.application_date.between(start_date, end_date), Transaction.deleted==None)
        actualExpenses = - (tq.filter(Transaction.value < 0).scalar() or Decimal('0'))
        actualIncomes = tq.filter(Transaction.value >= 0).scalar() or Decimal('0')

        balance = query(func.sum(Account.balance), Account).scalar() or Decimal('0')
        anticipated_balance = balance + (query(func.sum(Transaction.value)).filter(Transaction.coming == True, Transaction.rdate < end_date, Transaction.deleted==None).scalar() or Decimal('0'))

        income = self.compute_projection(query, start_date, end_date, True)
        outcome = self.compute_projection(query, start_date, end_date, False)

        balanceL = anticipated_balance + income['catsum'] + income['predsumL'] - outcome['catsum'] - outcome['predsumL']
        balanceH = anticipated_balance + income['catsum'] + income['predsumH'] - outcome['catsum'] - outcome['predsumH']
        balanceA = anticipated_balance + income['catsum'] + income['predsumA'] - outcome['catsum'] - outcome['predsumA']

        restL = (actualIncomes + income['catsum'] + income['predsumL']) - (actualExpenses + outcome['catsum'] + outcome['predsumL'])
        restH = (actualIncomes + income['catsum'] + income['predsumH']) - (actualExpenses + outcome['catsum'] + outcome['predsumH'])
        restA = (actualIncomes + income['catsum'] + income['predsumA']) - (actualExpenses + outcome['catsum'] + outcome['predsumA'])

        response = {'actualExpenses': - int(round(actualExpenses)),
                    'anticipatedExpenses': - int(round(outcome['sum'])),
                    'actualIncomes': int(round(actualIncomes)),
                    'anticipatedIncomes': int(round(income['sum'])),
                    'actualBalance': int(round(anticipated_balance)),
                    'anticipatedBalanceMin': int(round(balanceL)),
                    'anticipatedBalanceMax': int(round(balanceH)),
                    'anticipatedBalance': int(round(balanceA)),
                    'actualResult': int(round(actualIncomes - actualExpenses)),
                    'anticipatedResultMin': int(round(restL)),
                    'anticipatedResultMax': int(round(restH)),
                    'anticipatedResult': int(round(restA)),
                    'anticipatedVarExpensesMin': - int(round(outcome['predsumL'])),
                    'anticipatedVarExpensesMax': - int(round(outcome['predsumH'])),
                    'anticipatedVarExpenses': - int(round(outcome['predsumA'])),
                    'totalFixedExpenses': - int(round(outcome['catsum'])),
                    'expenses': self.format_clusters(outcome['categories']),
                    'anticipatedVarIncomesMin': int(round(income['predsumL'])),
                    'anticipatedVarIncomesMax': int(round(income['predsumH'])),
                    'anticipatedVarIncomes': int(round(income['predsumA'])),
                    'totalFixedIncomes': int(round(income['catsum'])),
                    'incomes': self.format_clusters(income['categories']),
                    'anticipatedValuesDate': end_date,
                    'balances': [],
                   }

        start_date = today - timedelta(days=2*30)
        end_date = today + timedelta(days=1*30)
        predictions = {r[0]: {'mean_amount': r[1], 'std_amount': r[2]} for r in query(Prediction.day, func.sum(Prediction.mean_amount), func.sum(Prediction.std_amount)).group_by(Prediction.day)}

        date = start_date
        while date <= end_date:
            data = {}
            data['date'] = date
            data['value'] = balance - (query(func.sum(Transaction.value)).filter(Transaction.rdate > date, Transaction.coming == False, Transaction.deleted==None).scalar() or Decimal('0'))
            data['value'] += query(func.sum(Transaction.value)).filter(Transaction.coming == True, Transaction.rdate < date, Transaction.deleted==None).scalar() or Decimal('0')
            if date > today:
                bmin = bmax = data['value']
                d = today
                last_pred = None
                while d < date:
                    try:
                        pred = predictions[d.day-1]
                    except KeyError:
                        pred = {'mean_amount': Decimal('0.0'), 'std_amount': Decimal('0.0')}

                    if last_pred is not None:
                        if d.day != 1:
                            bmin += (last_pred['mean_amount'] - pred['mean_amount']) - (last_pred['std_amount'] - pred['std_amount'])
                            bmax += (last_pred['mean_amount'] - pred['mean_amount']) + (last_pred['std_amount'] - pred['std_amount'])
                    last_pred = pred

                    for lines in chain(outcome['categories'].itervalues(), income['categories'].itervalues()):
                        for line in lines:
                            if not line['tdate']:
                                continue

                            if not line['disabled'] and not line['debited'] and line['tdate'] == d:
                                bmin += line['value']
                                bmax += line['value']
                    d += timedelta(days=1)

                data['value'] = (bmin + bmax) / 2

            data['value'] = int(round(data['value']))

            response['balances'].append(data)
            date += timedelta(days=2)

        first_date, last_date = query(func.min(Transaction.date), func.max(Transaction.date)).one()
        if not all([first_date, last_date]) or (last_date - first_date).days < 90:
            response['warning'] = True

        return JsonResponse(response)

    def format_clusters(self, categories):
        operations = []
        for id, ops in categories.iteritems():
            if not id:
                id = 9998

            for op in ops:
                if not op['tdate']:
                    continue

                no = {'wording':    op['text'],
                      'value':      int(round(op['cluster'].mean_amount)),
                      'date':       op['tdate'],
                      'done':       op['debited'],
                      'category':   None,
                     }
                if op['category']:
                    no['category'] = {'id':     op['category'].id,
                                      'name':   op['category'].name,
                                      'color':  op['category'].color,
                                     }
                else:
                    no['category'] = {'id':     9998,
                                      'name':   u'Indéfini',
                                      'color':  'D7D3BC',
                                     }
                operations.append(no)

        operations.sort(key=lambda no: no['date'])
        return operations



class UnplannedAlert(UserAlert):
    SUBJECT = u'%(app_name)s vous suggère d\'effectuer une dépense !'
    TEMPLATE = 'alert_unplanned'

    def __init__(self, profile, cluster):
        UserAlert.__init__(self, profile)
        self.vars['cluster'] = cluster


class ForecastsPlugin(Plugin):
    def cb_update_clusters(self, worker, job, session):
        for acc in session.query(Account).filter((Account.id_user==job.data['id'])&(Account.disabled==None)):
            self.update_forecasts(acc)
        return {}

    def update_forecasts(self, acc):
        session = object_session(acc)

        start_date = session.query(func.min(Transaction.rdate)).filter(Transaction.id_account==acc.id).one()[0]
        if start_date is None:
            return

        # Start from a full month.
        if start_date.day > 4:
            start_date = next_month(start_date)

        start_date = max(start_date, (datetime.date.today() - timedelta(days=6*31)).replace(day=1))

        records = acc.transactions.filter(Transaction.rdate>=start_date).order_by(Transaction.rdate).all()

        if len(records) == 0:
            return

        if not session.domain in self.main_cats:
            self.build_categories_cache(session)

        # Calculate clusters
        clusterer = TransactionClusterer(records, self.main_cats[session.domain])
        clusterer.find_clusters(acc, datetime.date.today())

        # do not consider unplanned transactions
        for c in acc.clusters.filter(TransactionsCluster.next_date!=None):
            #self.logger.debug('%%%% update_projection iteration on cluster %s' % c.wording.encode('utf-8', 'replace'))
            cluster_records = c.transactions
            ids = [tr.id for tr in cluster_records]
            cluster = clusterer.find_cluster(ids)
            if cluster is not None:
                #self.logger.debug('cluster found')
                if cluster.add_records(cluster_records) >= 1 and cluster.next_date <= datetime.date.today():
                    # if the cluster was not full, we should retry to find any missing transaction
                    cluster.find_missing_transaction(acc, datetime.date.today())
                    cluster.refresh()
            elif c.enabled:
                #self.logger.debug('cluster not found')
                cluster = clusterer.add_old_cluster(c, cluster_records, acc, datetime.date.today())

            if cluster is None:
                c.enabled = False
            else:
                clusterer.clusters.remove(cluster)
                if c.enabled:
                    c.mean_amount = cluster.mean_amount
                    c.median_increment = cluster.median_increment
                    c.next_date = cluster.next_date
                    c.wording = cluster.wording
                    c.id_category = cluster.category_id

                    for tr in cluster.records:
                        if tr.id_cluster is None:
                            tr.id_cluster = c.id

        for cluster in clusterer.clusters:
            c = TransactionsCluster(id_account=acc.id, mean_amount=cluster.mean_amount, median_increment=cluster.median_increment, next_date=cluster.next_date, wording=cluster.wording, id_category=cluster.category_id)
            session.add(c)
            session.flush()
            for record in cluster.records:
                record.id_cluster = c.id

        # Check if there is enough data to calculate prediction.
        count = session.query(func.count('*').label('nb')).filter(Transaction.id_account==acc.id).group_by(func.year(Transaction.rdate), func.month(Transaction.rdate))
        avg = session.query(func.avg(count.subquery().columns.nb)).scalar()
        if avg is None or avg < 10:
            for d in xrange(31):
                session.merge(Prediction(id_account=acc.id, day=d))
        else:
            # Calculate prediction
            prediction = PredictionCalculator()
            prediction.add_transactions(records, ignore_after=datetime.date.today().replace(day=1))
            prediction.compute_averages()
            for d in xrange(31):
                mean_amount, std_amount = prediction.get_prediction_still_to_be_spent(d+1)
                session.merge(Prediction(id_account=acc.id, day=d, mean_amount=mean_amount, std_amount=std_amount))


    def cb_check_projected_transaction(self, worker, job, session):
        value = - abs(Decimal(job.data['value']))
        if 'date' in job.data:
            date = parse_date(job.data['date']).date()
        else:
            date = None

        try:
            acc = session.query(Account).filter(Account.id==job.data['id_account']).one()
        except NoResultFound:
            return {'error': 'notfound'}

        res = self.check_unplanned_transaction(acc, value, date)

        COD2TXT = {self.PLAN_OK: 'ok',
                   self.PLAN_WARN: 'warning',
                   self.PLAN_NO: 'no',
                   self.PLAN_UNKNOWN: 'unknown',
                  }

        return {'result': COD2TXT[res]}

    def check_unplanned_transactions(self, acc):
        for cluster in acc.clusters.filter(TransactionsCluster.enabled==True, TransactionsCluster.next_date==None):
            if self.check_unplanned_transaction(acc, cluster.mean_amount) == self.PLAN_OK:
                cluster.next_date = datetime.date.today() + timedelta(days=1)

                for profile in acc.user.profiles:
                    report = UnplannedAlert(profile, cluster)
                    report.track(profile, 'unplanned')
                    report.send()

    PLAN_OK = 2
    PLAN_WARN = 1
    PLAN_NO = 0
    PLAN_UNKNOWN = -1
    def check_unplanned_transaction(self, acc, value, planned_date=None):
        clusters = acc.clusters.filter(TransactionsCluster.enabled==True)
        transactions = acc.transactions.filter(Transaction.coming==True)
        predictions = acc.predictions

        today = datetime.date.today()
        if planned_date is not None and (planned_date - today) > timedelta(days=45):
            return self.PLAN_UNKNOWN

        user_alert = acc.user.user_alert
        steps = sorted([user_alert.balance_min1, user_alert.balance_min2])

        end = today + relativedelta(months=1)
        bmin = bmax = acc.balance
        last_pred = None
        result = self.PLAN_OK

        if planned_date is None:
            bmin += value
            bmax += value

        while today <= end:
            try:
                prediction = predictions[today.day-1]
            except IndexError:
                prediction = Prediction(mean_amount=0, std_amount=0)

            if last_pred is not None:
                bmin += (last_pred.mean_amount - prediction.mean_amount) - (last_pred.std_amount - prediction.std_amount)
                bmax += (last_pred.mean_amount - prediction.mean_amount) + (last_pred.std_amount - prediction.std_amount)
            last_pred = prediction
            for cluster in clusters:
                if cluster.next_date is None:
                    continue
                if cluster.next_date == today or cluster.next_date + timedelta(days=(cluster.median_increment or 0)) == today:
                    bmin += cluster.mean_amount
                    bmax += cluster.mean_amount
            for tr in transactions:
                if tr.date == today:
                    bmin += tr.value
                    bmax += tr.value
            if planned_date == today:
                bmin += value
                bmax += value

            if bmin <= steps[1] and result >= self.PLAN_OK:
                result = self.PLAN_WARN
            if bmin <= steps[0] and result >= self.PLAN_WARN:
                result = self.PLAN_NO

            today += timedelta(days=1)

        return result

    enable_config_key = 'forecasts.enabled'

    def build_categories_cache(self, session):
        self.main_cats[session.domain] = {}
        for cat in session.query(Category):
            self.main_cats[session.domain][cat.id] = cat.id_parent_category

    def init(self, config):
        self.main_cats = {}

        self.register_hook('account_transactions_synced', self.update_forecasts)
        self.register_hook('account_transactions_synced', self.check_unplanned_transactions)
        self.register_gearman_command('update_clusters', self.cb_update_clusters)
        self.register_gearman_command('check_projected_transaction', self.cb_check_projected_transaction)
