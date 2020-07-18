import pandas as pd
import numpy as np
from statsmodels.multivariate.manova import MANOVA
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef,recall_score,roc_auc_score,roc_curve,confusion_matrix
from boxjenk import boxjenk as bj
from boxjenk2 import boxjenk as bj2
from boxjenk3 import boxjenk as bj3
from boxjenk4 import boxjenk as bj4


class ycrandom():
      def __init__(self,df,nc,desc,ni,model,no):
            self.df=df
            self.nc=nc
            self.desc=desc
            self.ni=ni
            self.model=model
            self.no=no
      
      def shuffling(self,df, n=1, axis=0):     
          df = df.copy()
          for _ in range(n):
              df.apply(np.random.shuffle, axis=axis)
          return df
      
      def cal_1(self,df,c):
          bjk=bj(df,df,c)
          i,j=bjk.fit()
          return i
      
      def cal_2(self,df,c):
          bjk=bj2(df,df,c)
          i,j=bjk.fit()
          return i
      
      def cal_3(self,df,c):
          bjk=bj3(df,df,c)
          i,j=bjk.ncal()
          return i

      def cal_4(self,df,c):
          bjk=bj4(df,df,c)
          i,j=bjk.ncal()
          return i

      
      def randomization(self):
          C,A,X=[],[],[]
          for i in range(0,self.ni):
              inc=self.df.iloc[:,0:1]
              yr=self.shuffling(self.df.iloc[:,1:2])
              #yr=self.df.iloc[:,1:2]
              c=self.nc
              cr=self.shuffling(self.df.iloc[:,2:c+2])
              xr=self.df.iloc[:,c+2:]
              ndr=pd.concat([inc,yr,cr,xr],axis=1)
              #ndr.to_csv('ndr.csv')
              if self.no==1:
                 dfbjr=self.cal_1(ndr.iloc[:,0:-1],c)
              elif self.no==2:
                 dfbjr=self.cal_2(ndr.iloc[:,0:-1],c)
              elif self.no==3:
                 dfbjr=self.cal_3(ndr.iloc[:,0:-1],c)
              elif self.no==4:
                 dfbjr=self.cal_4(ndr.iloc[:,0:-1],c)
              #dfbjr.to_csv('dfbjr.csv', index=False)
              s=self.df.iloc[:,-1:]
              dfbjr=pd.concat([dfbjr,s],axis=1)
              #dfbjr.to_csv('sfd2.csv',index=False)
              dfbjtr=dfbjr[dfbjr['Set']=='Sub_train']
              #dfbjtr.to_csv('sfd.csv',index=False)
              xrd=dfbjtr[self.desc]
              yr=dfbjtr[yr.columns]
              table=MANOVA.from_formula('xrd.values~ yr.values',data=dfbjtr).mv_test().results['yr.values']['stat']
              self.model.fit(xrd,yr)
              ypr=self.model.predict(xrd)
              acc=accuracy_score(yr,ypr)*100
          C.append(table.iloc[0,0])
          A.append(np.mean(acc))
          return C,A