from pyspark import SparkContext
from pyspark.ml.recommendation import ALS
from pyspark.sql import SQLContext
from RecommenderSystem.param import *

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

    def __init__(self,ranks=None,model='tucker',lbda=0.8,alpha=0.1,maxIter=5,
                dimensions_col=['user','item','action'],rating_col='rating',
                implicitPrefs=False,seed=None):
        """
        Parameters
        ranks: list of integer
        model: string
            either 'tucker' or 'cp' (case insensitive)
        """

        # context
        self.dimensions_col = dimensions_col
        self.rating_col = rating_col
        self.seed = seed
        self.nDim = len(dimensions_col)
        self.model = model

        # paramMap
        self.paramMap = recoParamMap()

        # all params 
        lbda_ = HasLbda(self.paramMap)
        lbda_.setLbda(lbda)

        alpha_ = HasAlpha(self.paramMap)
        alpha_.setAlpha(alpha)

        maxIter_ = HasMaxIter(self.paramMap)
        maxIter_.setMaxIter(maxIter)

        self.implicitPrefs = implicitPrefs
    
        if model.lower()=='tucker':
            ranks_ = dict.fromkeys(dimensions_col)
            for dim in dimensions_col:
                ind = dimensions_col.index(dim)
                ranks_[dim] = HasRank(self.paramMap,dim)
                val = ranks[ind] if ranks is not None else 1
                ranks_[dim].setRank(val)
        else:
            rank_ = HasRank(self.paramMap,'rank',None)
            val = ranks if type(ranks) is int else ranks[0] if ranks is not None else 1
            rank_.setRank(val)
        
    def describe(self):
        """ 
        to do: align print based on length of words
        """
        print('-'*15+'> model','=',self.model)
        print('-'*7+'> implicitPrefs','=',self.implicitPrefs)
        
        self.paramMap.describe()

    def get(self,name,parent=None):
        return self.paramMap.get(name,parent)

    def set(self,name,value,parent=None,desc=None):
        param_ = recoParam(parent,name,desc,value)
        self.paramMap.addOrUpdate(name,param_)

    def fit(self,tensor,timer=False):
        # add check that each dimensions_col start at 0
        self.tensor = tensor 

        self.dims = dict.fromkeys(self.dimensions_col)
        for col in self.dimensions_col:
            self.dims[col] = self.tensor.shape[self.dimensions_col.index(col)]

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
            print("\t Start "+mode+" learning")
            
            ind = self.dimensions_col.index(mode)
            local_dataset = sqlContext.createDataFrame(datas[mode])

            # Build the recommendation model using Alternating Least Squares
            if timer:
                t0 = time.time()
            
            if self.model=='tucker':
                rank = self.get(mode,'ranks')
            else:
                rank = self.get('rank',None)
            local_als = ALS(rank=rank,
                            maxIter=self.get('maxIter'),
                            regParam=self.get('lbda'),
                            alpha=self.get('alpha'),
                            implicitPrefs=self.implicitPrefs,
                            userCol='row',itemCol='col',ratingCol='rating',seed=self.seed)
            res[mode] = local_als.fit(local_dataset)
            if timer:
                t1 = time.time()
                delta = t1-t0
                print('\t \t time :',delta,"seconds")
                times.append(delta)

            latentFactors = res[mode].userFactors#.orderBy("id")
            latentFactors_index = latentFactors.select('id').toPandas()
            latentFactors = latentFactors.select('features')

            for k in range(rank):
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
            print('\t \t longest mode time :',np.max(times),"seconds")

        if self.model.lower()=="tucker":
            print("\t Get core tensor")
            # get W
            if self.implicitPrefs:
                self.tensor.vals = np.repeat(1,len(self.tensor.vals))
            
            self.W = deepcopy(self.tensor)
            for mode in self.dimensions_col:
                ind = self.dimensions_col.index(mode)
                self.W = self.W.ttm(np.linalg.pinv(self.features[mode]),mode=ind)
    
    def predict(self,indexes):
        if self.model.lower()=="tucker":
            # adapt formula for tensor of dimension higher than 3
            mode = self.dimensions_col[0]
            out = self.W.ttv(self.features[mode][indexes[0],:],modes=0).T

            for ind in range(1,len(self.dimensions_col)):
                mode = self.dimensions_col[ind]
                out = out.dot(self.features[mode][indexes[ind],:])
        else:
            out = 1
            rank = self.get('rank',None)
            for r in np.arange(rank):
                for mode in self.dimensions_col:
                    ind = self.dimensions_col.index(mode)
                    out *= self.features[mode][indexes[ind],r]
        return out

    def predict_old(self,indexes):
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
            rank = self.paramMap.get(self.dimensions_col[0],'ranks')
            out = np.ones(rank)
            for mode in self.dimensions_col:
                ind = self.dimensions_col.index(mode)
                out = np.multiply(out,self.features[mode][indexes[index][ind],:])
            out = np.sum(out)
        
        return out

    def getFullTensor(self):
        # warning: can raise memory issue. So do it only for very small tensor
        if self.model.lower()=="tucker":
            T_hat = self.W
            for mode in self.dimensions_col:
                ind = self.dimensions_col.index(mode)
                T_hat = T_hat.ttm(self.features[mode],mode=ind)
        return T_hat
