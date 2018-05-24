from pyspark import SparkContext
from pyspark.ml.recommendation import ALS
from pyspark.sql import SQLContext

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
import pandas as pd
import numpy as np
import sktensor
import re
import itertools
import time

class HOALS(object):

    def __init__(self,ranks,model='tucker',lbda=0.8,alpha=0.1,max_iter=5,
                dimensions_col=['user','item','action'],rating_col=['rating'],implicitPrefs=False):
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
        self.max_iter = max_iter
        self.dimensions_col = dimensions_col
        self.rating_col = rating_col
        self.implicitPrefs = implicitPrefs
        self.nDim = len(ranks)

    def fit(self,dataset,time=False):
        self.dims = dict.fromkeys(self.columns)

        shape = []
        for col in self.columns:
            self.dims[col] = len(set(dataset[col]))
            shape.append(self.dims[col])

        subs = []
        for col in self.columns:
            subs.append(list(data[col]))

        tensor = sptensor(tuple(subs), data[rating_col], shape=tuple(shape), dtype=np.float)

        #==============================================================================
        # recuparation of the (user,item,rate) of the unfold matrix
        #==============================================================================
        unfolded_matrix = dict.fromkeys(self.columns)
        datas = dict.fromkeys(self.columns)

        for col in self.columns:
            ind = self.columns.index(col)
            
            unfolded_matrix[col] = sktensor.csr_matrix(tensor.unfold(ind))
            y = list(unfolded_matrix[col].indices)
            indptr = unfolded_matrix[col].indptr
            r = list(unfolded_matrix[col].data)
            tmp = indptr[1:len(indptr)]-indptr[0:(len(indptr)-1)]
            x = []
            for i in np.arange(len(tmp)):
                x.extend(np.repeat(i,tmp[i]))

            datas[col] = pd.DataFrame([x,y,r],columns=['row','col','rating']).T

        #==============================================================================
        # Factorization
        #==============================================================================
        res = dict.fromkeys(self.columns)
        self.features = dict.fromkeys(self.columns)
        features_star = dict.fromkeys(self.columns)
        if self.time:
            times = []
        
        for mode in self.columns:
            print("Start "+mode+" Learning")
            
            ind = self.columns.index(mode)
            local_dataset = sqlContext.createDataFrame(datas[mode])

            # Build the recommendation model using Alternating Least Squares
            if self.time:
                t0 = time.time()
            
            local_als = ALS(rank=ranks[mode],maxIter=self.max_iter,regParam=self.lbda,alpha=self.alpha,implicitPrefs=self.implicitPrefs,
                            userCol='row',itemCol='col',ratingCol='rating')
            res[mode] = ALS.fit(local_dataset)
            if self.time:
                t1 = time.time()
                delta = t1-t0
                print('time :',delta)
                times.append(delta)

            self.features[mode] = res[mode].userFactors.orderBy("id")
        print('longest mode time :',np.max(times))

        if model.lower()=="tucker":
            # get W
            if self.implicitPrefs:
                tensor.vals = np.repeat(1,len(dataset))
            
            self.W = tensor.ttm(np.linalg.pinv(self.features[0]),mode=1)
            for mode in range(1,nDim-1):
                self.W = self.W.ttm(np.linalg.pinv(self.features[ind]),mode=ind+1)
            self.W = self.W.ttm(np.linalg.pinv(self.features[nDim-1]),mode=0)

    def predict(self,indexes):
        out = 0
        if model.lower()=="tucker":
            P = [range(self.ranks[k]) for k in range(len(ranks))]
            for elt in itertools.product(*P):
                m,l,s = elt
                tmp = self.W[s,m,l]
                for ind in range(self.nDim):
                    tmp *= self.features[ind][indexes[ind,m]]
                out += tmp
        else:
            out = 0
            for r in np.arange(ranks[0]):
                tmp = 1
                for ind in range(self.nDim):
                    tmp *= self.features[ind][indexes[ind,r]]
                
