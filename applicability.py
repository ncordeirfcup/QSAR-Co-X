import numpy as np
import pandas as pd
class apdom():
      def __init__(self,test,train):
            self.train=train
            self.test=test
      def zscore(self,data):
         x_,y_ = data.shape
         media = np.zeros(shape=(y_), dtype=np.float32)
         sigma = np.zeros(shape=(y_), dtype=np.float32)
    
         l,m=[],[]
         for j in range(y_):
             media[j] = data.iloc[:,j].mean()
             sigma[j] = data.iloc[:,j].std()
             l.append(media[j])
             m.append(sigma[j])
        
         result = np.copy(data)
         for i in range(x_):
             for j in range(y_):
                 result[i,j] = ((data.iloc[i,j] - media[j]) / sigma[j])
         return result,l,m
      def zscore_test(self,data,train):
          u,x,y=self.zscore(train)
          x_,y_ = data.shape
          media = np.zeros(shape=(y_), dtype=np.float32)
          sigma = np.zeros(shape=(y_), dtype=np.float32)

          for j in range(y_):
              media[j] = x[j]
              sigma[j] = y[j]
        
          result = np.copy(data)
          for i in range(x_):
              for j in range(y_):
                  result[i,j] = ((data.iloc[i,j] - media[j]) / sigma[j])
          return result
      def AD(self,X):
          x_,y_=X.shape
          #stX=pd.DataFrame(zscore(X))
          #stX=pd.DataFrame(preprocessing.scale(X))
          #stX=stX.abs()
          X['mean']=X.mean(axis=1)
          X['std']=X.iloc[:,0:-1].std(axis=1)
          X['snew']=X['std']*1.28+X['mean']
          X['Outlier_info(standardization_approach)']=['Outlier' if x>3 else 'In' for x in X['snew']]
          X=X.drop(['mean','std'],axis=1)
          return X
      def fit(self):
          stX=pd.DataFrame(self.zscore_test(self.test,self.train),columns=str('std')+self.test.columns)
          stX=stX.abs()
          stX['max']=stX.max(axis=1)
          stX['Outlier_info(standardization_approach)']=['Outlier' if x>3 else 'In' for x in stX['max']]
          stX2=stX[stX['Outlier_info(standardization_approach)']=='Outlier']
          stX1=stX[stX['Outlier_info(standardization_approach)']!='Outlier']
          stX1=stX1.drop(['max'],axis=1)
          stX2=stX2.drop(['Outlier_info(standardization_approach)','max'],axis=1)
          stX3=self.AD(stX2)
          xy=pd.concat([stX1,stX3],axis=0)
          xy=xy.sort_index()
          xy=pd.concat([xy],axis=1).fillna('NA')
          return xy     