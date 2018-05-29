from pyspark import SparkContext
from pyspark.ml.recommendation import ALS
from pyspark.sql import SQLContext

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
import pandas as pd
import numpy as np
from sktensor.sptensor import sptensor
from scipy.sparse import csr_matrix
import itertools
import time
from copy import deepcopy

class HOALS(object):

    def __init__(self,ranks,model='tucker',lbda=0.8,alpha=0.1,max_iter=5,
                dimensions_col=['user','item','action'],rating_col=['rating'],
                implicitPrefs=False):
        """
        Parameters
        ranks: list of integer
        model: string
            either 'tucker' or 'cp' (case insensitive)
        """

        self.ranks = ranks
        if model.lower()!='tucker':
            self.ranks = [ranks[0]]*len(ranks) # ensure that all dimensions are equal
        self.model = model
        self.lbda = lbda
        self.alpha = alpha
        self.max_iter = max_iter
        self.dimensions_col = dimensions_col
        self.rating_col = rating_col
        self.implicitPrefs = implicitPrefs
        self.nDim = len(ranks)

    def fit(self,dataset,timer=False):
        # add check that each dimensions_col start at 0
        self.dims = dict.fromkeys(self.dimensions_col)

        shape = []
        for col in self.dimensions_col:
            self.dims[col] = max(dataset[col])+1
            shape.append(self.dims[col])

        subs = []
        for col in self.dimensions_col:
            subs.append(list(dataset[col]))

        self.tensor = sptensor(tuple(subs), dataset[self.rating_col], shape=tuple(shape), dtype=np.float)

        #==============================================================================
        # recuparation of the (user,item,rate) of the unfold matrix
        #==============================================================================
        unfolded_matrix = dict.fromkeys(self.dimensions_col)
        datas = dict.fromkeys(self.dimensions_col)

        for dim in self.dimensions_col:
            ind = self.dimensions_col.index(dim)
            
            unfolded_matrix[dim] = csr_matrix(self.tensor.unfold(ind))
            y = list(unfolded_matrix[dim].indices)
            indptr = unfolded_matrix[dim].indptr
            r = list(unfolded_matrix[dim].data)
            tmp = indptr[1:len(indptr)]-indptr[0:(len(indptr)-1)]
            x = []
            for i in np.arange(len(tmp)):
                x.extend(np.repeat(i,tmp[i]))

            datas[dim] = pd.DataFrame({'row':x,'col':y,'rating':r})

        #==============================================================================
        # Factorization
        #==============================================================================
        res = dict.fromkeys(self.dimensions_col)
        self.features = dict.fromkeys(self.dimensions_col)
        features_star = dict.fromkeys(self.dimensions_col)
        if timer:
            times = []
        
        for mode in self.dimensions_col:
            print("Start "+mode+" Learning")
            
            ind = self.dimensions_col.index(mode)
            local_dataset = sqlContext.createDataFrame(datas[mode])

            # Build the recommendation model using Alternating Least Squares
            if timer:
                t0 = time.time()
            
            local_als = ALS(rank=self.ranks[ind],maxIter=self.max_iter,regParam=self.lbda,alpha=self.alpha,implicitPrefs=self.implicitPrefs,
                            userCol='row',itemCol='col',ratingCol='rating')
            res[mode] = local_als.fit(local_dataset)
            if timer:
                t1 = time.time()
                delta = t1-t0
                print('time :',delta)
                times.append(delta)

            latentFactors = res[mode].userFactors#.orderBy("id")
            latentFactors_index = latentFactors.select('id').toPandas()
            latentFactors = latentFactors.select('features')
            for k in range(self.ranks[ind]):
                latentFactors = latentFactors.withColumn('factor'+str(k),latentFactors.features[k])
            latentFactors = latentFactors.drop('features')
            latentFactors = latentFactors.toPandas()
            latentFactors.index = latentFactors_index['id']
            unknowns = list(set(range(self.dims[mode]))-set(latentFactors_index['id']))
            for unknown in unknowns:
                latentFactors.loc[unknown] = 0
            latentFactors = latentFactors.sort_index()
            self.features[mode] = np.array(latentFactors)
        if timer:
            print('longest mode time :',np.max(times))

        if self.model.lower()=="tucker":
            # get W
            if self.implicitPrefs:
                self.tensor.vals = np.repeat(1,len(dataset))
            
            self.W = deepcopy(self.tensor)
            for mode in self.dimensions_col:
                ind = self.dimensions_col.index(mode)
                self.W = self.W.ttm(np.linalg.pinv(self.features[mode]),mode=ind)
            
    def predict(self,indexes):
        # check if indexes is:
        # - list -> for one prediction
        # - list of list -> multiple prediction 
        indexes_size = len(indexes)
        out = np.zeros(indexes_size)
        if self.model.lower()=="tucker":
            P = [range(self.ranks[k]) for k in range(len(self.ranks))]
            for elt in itertools.product(*P):
                tmp = np.repeat(self.W[elt],indexes_size)
                for mode in self.dimensions_col:
                    ind = self.dimensions_col.index(mode)
                    for index in range(indexes_size):
                        tmp[index] *= self.features[mode][indexes[index][ind],elt[ind]]
                out += tmp
        else:
            out = np.ones(indexes_size)
            for r in np.arange(ranks[0]):
                for mode in self.dimensions_col:
                    ind = self.dimensions_col.index(mode)
                    for index in range(indexes_size):
                        out[index] *= self.features[ind][indexes[index][ind],r]
        
        return out

    def getFullTensor(self):
        # warning: can raise memory issue. So do it only for very small tensor
        if self.model.lower()=="tucker":
            T_hat = self.W
            for mode in self.dimensions_col:
                ind = self.dimensions_col.index(mode)
                T_hat = T_hat.ttm(self.features[mode],mode=ind)
        return T_hat
