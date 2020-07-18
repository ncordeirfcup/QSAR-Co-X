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
          kmeans = KMeans(n_clusters=self.nclus) 
          kmeans.fit(X)
          ncol=pd.DataFrame(kmeans.labels_, columns=['cluster'])
          dfn=pd.concat([self.df,ncol], axis=1)
          #dfn.to_csv('dfn.csv',index=False)
          m=[]
          for i in dfn['cluster'].unique():
              di=dfn[dfn['cluster']==i]
              if di.shape[0]==1:
                 di['Set']=='Remove'
                 m.append(di)
              elif di.shape[0]>1:
                  ai,bi= train_test_split(di,test_size=self.dev, random_state=self.seed)
                  ai['Set']='Train'
                  bi['Set']='Test'
                  dn=pd.concat([ai,bi],axis=0)
                  dn=dn[[self.df.iloc[:,0:1].columns[0],'Set']]
                  m.append(dn)
                  ad=pd.concat(m,axis=0)
          nd=pd.merge(self.df,ad,on=self.df.iloc[:,0:1].columns[0],how='left')
          nd.to_csv('nd.csv',index=False)
          tr=nd[nd['Set']=='Train']
          tr=tr.drop(['Set'],axis=1)
          ts=nd[nd['Set']=='Test']
          ts=ts.drop(['Set'],axis=1)
          return tr,ts
          