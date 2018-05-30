import numpy as np
import pandas as pd

def crossval_split(data,userPerc,ratingsPerc,userCol,itemCol):
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
	"""
	nUser = len(set(data[userCol]))
	
	usersItems = data[[userCol,itemCol]].drop_duplicates()

	trainingUsers = np.random.choice(nUser,int(userPerc*nUser),replace=False)
	testUsers = list(set(range(nUser))-set(trainingUsers))
	trainingUsers = pd.DataFrame({userCol:trainingUsers})
	testUsers = pd.DataFrame({userCol:testUsers})

	trainingData = pd.merge(trainingUsers,data) # all information for those users
	testUsersItems = pd.merge(testUsers,usersItems) # only user / item couple

	testUsersItems['cv'] = np.random.random(len(testUsersItems))
	testUsersItems_train = testUsersItems.loc[testUsersItems['cv']<ratingsPerc] # user / item couple to be add to the trainingData
	testUsersItems_test = testUsersItems.loc[testUsersItems['cv']>=ratingsPerc] # user / item couple to be form the testData

	del testUsersItems_train['cv']
	del testUsersItems_test['cv']
	testData_train = pd.merge(testUsersItems_train,data)
	testData = pd.merge(testUsersItems_test,data)

	trainingData = pd.concat([trainingData,testData_train])

	return trainingData, testData