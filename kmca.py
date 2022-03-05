from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split

class kmca():
      def __init__(self,df,nc,dev,seed,nclus):
          self.df=df
          self.dev=dev
          self.seed=seed
          self.nclus=nclus
          self.nc=nc
      def cal(self):
          X=self.df.iloc[:,(self.nc+2):]
          kmeans = KMeans(n_clusters=self.nclus, random_state=42) #modified to fix the data distributions
          kmeans.fit(X)
          ncol=pd.DataFrame(kmeans.labels_, columns=['cluster'])
          dfn=pd.concat([self.df,ncol], axis=1)
          #dfn.to_csv('dfn.csv',index=False)
          m,n=[],[]
          for i in dfn['cluster'].unique():
              di=dfn[dfn['cluster']==i]
              if di.shape[0]==1:
                 di['Set']='Train'
                 m.append(di)
                 ad1=pd.concat(m,axis=0)
              elif di.shape[0]>1:
                  ai,bi= train_test_split(di,test_size=self.dev, random_state=self.seed)
                  ai['Set']='Train'
                  bi['Set']='Test'
                  dn=pd.concat([ai,bi],axis=0)
                  dn=dn[[self.df.iloc[:,0:1].columns[0],'Set']]
                  n.append(dn)
                  ad2=pd.concat(n,axis=0)
          nd=pd.merge(self.df,ad2,on=self.df.iloc[:,0:1].columns[0],how='left')
          if len(m)>0:
             ad3=pd.concat([nd,ad1],axis=0)
             ad3=ad3.drop(['cluster'],axis=1)
          else:
             ad3=nd
          #nd.to_csv('nd.csv',index=False)
          tr=ad3[ad3['Set']=='Train']
          tr=tr.drop(['Set'],axis=1)
          ts=ad3[ad3['Set']=='Test']
          ts=ts.drop(['Set'],axis=1)
          return tr,ts
          
