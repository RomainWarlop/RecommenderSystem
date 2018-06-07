import numpy as np
import pandas as pd
from sktensor.sptensor import sptensor
from pyspark.sql.functions import rand
from itertools import product as itProduct
from RecommenderSystem.param import *
from RecommenderSystem.metrics import rank_measure
from copy import deepcopy
from pyspark import SparkContext

sc = SparkContext.getOrCreate()

# to do:
# - gridsearch cv
# - adaptive gridsearch cv (start with set of parameters then remove some and add new until some criterion)

def traintest_split(dataset,userPerc,ratingsPerc,userCol,itemCol,seed=None):
	"""
	Perform a train / test split of the data set. 
	some users are totally in the training set
	other users are partially in the test. 
	If an item for a particular user is in the training (resp. test) set, all dimensions information are in the training (resp. test) set

	dataset: pyspark DataFrame
		contains user, item, supplementary dimension(s) and rating
	userPerc: float between 0 and 1
		percentage of users to keep in full in the training part
	ratingsPerc: float between 0 and 1
		percentage of ratings to keep in average for user in the test set
	userCol: string
		name of the user column
	itemCol: string
		name of the item column
	"""
	users = dataset.select(userCol).distinct()
	userRandCol = userCol+"_cv"
	users = users.select("*", rand(seed).alias(userRandCol))
	df = dataset.join(users,on=userCol)

	# rating subset
	ratingRandCol = "rating_cv"
	df = df.select("*", rand(seed).alias(ratingRandCol))

	condition = (df[userRandCol] <= userPerc) | ((df[userRandCol] > userPerc) & (df[ratingRandCol] <= ratingsPerc))
	trainingData = df.filter(condition).cache()
	testData = df.filter(~condition).cache()

	return trainingData, testData

def traintest_split_old(data,userPerc,ratingsPerc,userCol,itemCol,seed=None,split='global'):
	"""
	Perform a train / test split of the data set. 
	some users are totally in the training set
	other users are partially in the test. 
	If an item for a particular user is in the training (resp. test) set, all dimensions information are in the training (resp. test) set

	data: pandas DataFrame
		contains user, item, supplementary dimension(s) and rating
	userPerc: float between 0 and 1
		percentage of users to keep in full in the training part
	ratingsPerc: float between 0 and 1
		percentage of ratings to keep in average for user in the test set
	userCol: string
		name of the user column
	itemCol: string
		name of the item column
	split: string
		either 'global' or 'exact'. 
			- 'global': in average ratingsPerc percent of ratings are kept by each test users
			- 'exact': all test users have (almost) exactly ratingsPerc percent of ratings kept (use percentile condition)
	"""
	if seed is not None:
		np.random.seed(int(seed))

	nUser = len(set(data[userCol]))
	
	usersItems = data[[userCol,itemCol]].drop_duplicates()

	trainingUsers = np.random.choice(nUser,int(userPerc*nUser),replace=False)
	testUsers = list(set(range(nUser))-set(trainingUsers))
	trainingUsers = pd.DataFrame({userCol:trainingUsers})
	testUsers = pd.DataFrame({userCol:testUsers})

	trainingData = pd.merge(trainingUsers,data) # all information for those users
	testUsersItems = pd.merge(testUsers,usersItems) # only user / item couple

	if split=='exact':
		testUsersItems['cv'] = testUsersItems.groupby(userCol).transform(lambda x: np.random.random(len(x))) # random per user
		testUsersItems['t'] = testUsersItems.groupby(userCol)['cv'].transform(lambda x: np.percentile(x,ratingsPerc*100))
		testUsersItems_train = testUsersItems.loc[testUsersItems['cv']<testUsersItems['t']] # user / item couple to be add to the trainingData
		testUsersItems_test = testUsersItems.loc[testUsersItems['cv']>=testUsersItems['t']] # user / item couple to be form the testData
	else:
		testUsersItems['cv'] = np.random.random(len(testUsersItems)) # global random
		testUsersItems_train = testUsersItems.loc[testUsersItems['cv']<ratingsPerc] # user / item couple to be add to the trainingData
		testUsersItems_test = testUsersItems.loc[testUsersItems['cv']>=ratingsPerc] # user / item couple to be form the testData

	del testUsersItems_train['cv']
	del testUsersItems_test['cv']
	testData_train = pd.merge(testUsersItems_train,data)
	testData = pd.merge(testUsersItems_test,data)

	trainingData = pd.concat([trainingData,testData_train])

	return trainingData, testData

def predict_tucker(core,matrices,indexes):

    # adapt formula for tensor of dimension higher than 3
    out = core.ttv(matrices[0][indexes[0],:],modes=0).T

    for ind in range(1,len(matrices)):
        out = out.dot(matrices[ind][indexes[ind],:])

    return out

def predict_parafac(matrices,indexes):

    rank = matrices[0].shape[1]
    out = np.ones(rank)
    for ind in range(len(matrices)):
        out = np.multiply(out,np.array(matrices[ind])[indexes[ind],:])
    out = np.sum(out)
    return out

def dataframeToTensor(dataset,dimensions_col=['user','item','action'],rating_col='rating',keepZero=False):
    dims = dict.fromkeys(dimensions_col)

    shape = []
    for col in dimensions_col:
        dims[col] = max(dataset[col])+1
        shape.append(dims[col])

    if not(keepZero):
        dataset = dataset.loc[dataset[rating_col]!=0]

    subs = []
    for col in dimensions_col:
        subs.append(list(dataset[col]))

    tensor = sptensor(tuple(subs), dataset[rating_col].values, shape=tuple(shape), dtype=np.float)

    return tensor

def getRankMeasure(model,data):
	# check for generalization
	indexes = data[model.dimensions_col].values.tolist()
	r_hat = sc.parallelize(indexes).map(model.predict).collect()
	data['r_hat'] = r_hat

	dims = model.dimensions_col[2:]
	distinctDims = dict.fromkeys(dims)
	condition = []
	for dim in dims:
		distinctDims[dim] = set(data[dim])
		condition.append(list(distinctDims[dim]))

	rank_perf = pd.DataFrame(columns=dims+['metric'])
	for elt in itProduct(*condition):
		subdata = deepcopy(data)
		for col_index in range(len(elt)):
			subdata = subdata.loc[data[dims[col_index]]==elt[col_index],]
		rs = list(subdata.sort_values(by='r_hat',ascending=False).groupby(model.dimensions_col[0])[model.rating_col].apply(list))
		rank_perf.loc[len(rank_perf)] = list(elt)+[np.round(100*rank_measure(rs),2)]

	return rank_perf

class RecoParamGrip(object):

	def __init__(self):
		self.params = []

	def addGrid(self,parent,name,desc,values):
		self.params.append({'parent':parent,'name':name,'desc':desc,'values':values,'size':len(values)})

class RecoCrossValidator(object):
    
	def __init__(self, estimator=None, paramGrid=None, evaluator=None,
	             userPerc=0.8, ratingsPerc=0.7,userCol='user',itemCol='item', 
	             seed=None, parallelism=1):

		self.estimator = estimator
		self.paramGrid = paramGrid
		self.evaluator = evaluator
		self.userPerc = userPerc
		self.ratingsPerc = ratingsPerc
		self.userCol = userCol
		self.itemCol = itemCol
		self.seed = seed
		self.parallelism = parallelism

	def _fit(self, dataset, nFolds=2):
		cvParameters_ = []
		for elt in self.paramGrid.params:
			cvParameters_.append(list(range(elt['size'])))
		cvParameters = itProduct(*cvParameters_)
		
		columns = []
		for elt in self.paramGrid.params:
			if elt['parent'] is None:
				columns.append(elt['name'])
			else:
				columns.append(elt['parent']+'_'+elt['name'])

		metrics = []

		#pool = ThreadPool(processes=min(self.getParallelism(), numModels))
		for fold in range(nFolds):
			print('fold',fold+1,'/',nFolds)
			if self.seed is None:
				seed = None
			else:
				seed = self.seed+fold
			
			self.estimator.seed = seed

			train, validation = traintest_split(dataset,self.userPerc,self.ratingsPerc,self.userCol,self.itemCol,seed=seed)
			train = train.toPandas()
			validation = validation.toPandas()

			tensor = dataframeToTensor(train,dimensions_col=self.estimator.dimensions_col,
				rating_col=self.estimator.rating_col,keepZero=False)
			
			# to do in parallel ? -> https://www.timlrx.com/2018/04/08/creating-a-custom-cross-validation-function-in-pyspark/
			for elt in cvParameters:
				names = []
				values = []
				# loop over all parameter in paramGrid and select value corresponding to elt
				for i in range(len(self.paramGrid.params)):
					param = recoParam(parent=self.paramGrid.params[i]['parent'],
									  name=self.paramGrid.params[i]['name'],
									  desc=self.paramGrid.params[i]['name'],
									  value=self.paramGrid.params[i]['values'][elt[i]])
					name_ = param.name if param.parent is None else param.parent

					# modify param
					self.estimator.paramMap.addOrUpdate(name_,param)
					
					if param.parent is None:
						names.append(param.name)
					else:
						names.append(param.parent+'_'+param.name)
					values.append(self.paramGrid.params[i]['values'][elt[i]])

				# print current parameters
				self.estimator.describe()
				# fit model
				self.estimator.fit(tensor)
				metric = self.evaluator(self.estimator,validation)
				metric['fold'] = fold
				for i in range(len(names)):
					metric[names[i]] = values[i]

				metrics.append(metric)

			del validation
			del train

		return metrics