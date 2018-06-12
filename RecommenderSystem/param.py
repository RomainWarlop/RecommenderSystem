class recoParamMap(object):

	def addOrUpdate(self,name,param):
		if param.parent is None:
			setattr(self,name,param)
		else:
			try:
				parent = getattr(self,param.parent)
			except:
				setattr(self,param.parent,{})
				parent = getattr(self,param.parent)
			parent[param.name] = {'desc':param.desc,'value':param.value}

	def get(self,name,parent=None):
		if parent is None:
			out = getattr(self,name)
			if type(out)==dict:
				tmp = {}
				for key in out.keys():
					tmp[key] = out[key]['value']
				out = tmp
			else:
				out = out.value
		else:
			out = getattr(self,parent)
			out = out[name]['value']
		return out

	def describe(self):
		min_len = 20

		for key in self.__dict__.keys():
			elt = self.__dict__[key]
			if type(elt)==dict:
				for sub_key in elt.keys():
					print('-'*(min_len-len(key)-len(sub_key)-3)+'> '+key,'-',sub_key,'=',elt[sub_key]['value'])
			else:
				name = elt.name
				value = elt.value
				print('-'*(min_len-len(name))+'> '+name,'=',value)

class recoParam(object):

	def __init__(self,parent,name,desc,value=None):
		setattr(self,'parent',parent)
		setattr(self,'name',name)
		setattr(self,'desc',desc)
		setattr(self,'value',value)

class HasLbda(recoParam):

	def __init__(self,paramMap):
		super(HasLbda,self).__init__(None,'lbda','Tikhonov regularization weight',0)
		self.paramMap = paramMap
		self.paramMap.addOrUpdate('lbda',self)

	def setLbda(self,value):
		self.value = value
		self.paramMap.addOrUpdate('lbda',self)

	def getLbda(self):
		return self.value

class HasAlpha(recoParam):

	def __init__(self,paramMap):
		super(HasAlpha,self).__init__(None,'alpha','implicit weight',0)
		self.paramMap = paramMap
		self.paramMap.addOrUpdate('alpha',self)

	def setAlpha(self,value):
		self.value = value
		self.paramMap.addOrUpdate('alpha',self)

	def getAlpha(self):
		return self.value

class HasRank(recoParam):

	def __init__(self,paramMap,name,parent='ranks'):
		super(HasRank,self).__init__(parent,name,'rank value in '+name+' dimension',1)
		self.paramMap = paramMap
		if parent is None:
			paramMap.addOrUpdate(name,self)
		else:
			paramMap.addOrUpdate(parent,self)

	def setRank(self,value):
		self.value = value
		if self.parent is None:
			self.paramMap.addOrUpdate(self.name,self)
		else:
			self.paramMap.addOrUpdate(self.parent,self)
	
	def getRank(self):
		return self.value

class HasMaxIter(recoParam):

	def __init__(self,paramMap):
		super(HasMaxIter,self).__init__(None,'maxIter','maximum number of iterations',1)
		self.paramMap = paramMap
		self.paramMap.addOrUpdate('maxIter',self)

	def setMaxIter(self,value):
		self.value = value
		self.paramMap.addOrUpdate('maxIter',self)

	def getMaxIter(self):
		return self.value