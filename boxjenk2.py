import pandas as pd
from checkdata import check_data
from sklearn.feature_selection import VarianceThreshold

class boxjenk():
      def __init__(self,dftr,dfts,nc):
            self.dftr=dftr
            self.dfts=dfts
            self.nc=nc
      
      def variance(self,X,threshold):
          sel = VarianceThreshold(threshold=(threshold* (1 - threshold)))
          sel_var=sel.fit_transform(X)
          X=X[X.columns[sel.get_support(indices=True)]]    
          return X
        

      def pretreat(self):
          yc=self.dftr.iloc[:,0:(self.nc+2)]
          X=self.dftr.iloc[:,(self.nc+2):]
          X1=self.variance(X,0.0001)
          dftrpt=pd.concat([yc,X1],axis=1)
          dftr=self.dftr[dftrpt.columns]   
          dfts=self.dfts[dftrpt.columns]   
          return dftr,dfts 
 
      def calculation(self,df,dfts,n,nf):
          ci=pd.DataFrame(df.iloc[:,n])
          hi=ci.columns.values.tolist()
          dff=df[df.iloc[:,1]==1]
          dffs=dff.iloc[:,2:]
          dff2=dffs.groupby(hi).mean()
          dff4=pd.merge(df,dff2, on=hi, how='left',suffixes=('?', '!')).fillna(0)
          dff4.columns=dff4.columns.str.rstrip('?')
          dff4.columns=dff4.columns.str.rstrip('!')
          dff4ts=pd.merge(dfts,dff2, on=hi, how='left',suffixes=('?', '!')).fillna(0)
          dff4ts.columns=dff4ts.columns.str.rstrip('?')
          dff4ts.columns=dff4ts.columns.str.rstrip('!')
          #for training set
          fc=nf+2
          dff5=dff4.iloc[:,fc:]
          x_,y_=dff5.shape
          a=y_*0.5
          a=int(a)
          ldf=dff5.iloc[:,0:a]
          rdf=dff5.iloc[:,a:]
          li=[]
          for j in ldf:
              x=(ldf[j]-rdf[j])/(ldf[j].max()-ldf[j].min())
              li.append(x)
          trd=pd.DataFrame(li)
    #for test set
          fc=nf+2
          dff5ts=dff4ts.iloc[:,fc:]
          x_,y_=dff5ts.shape
          a=y_*0.5
          a=int(a)
          ldfts=dff5ts.iloc[:,0:a]
          rdfts=dff5ts.iloc[:,a:]
          lits=[]
          for j in ldf:
              xts=(ldfts[j]-rdfts[j])/(ldf[j].max()-ldf[j].min())
              lits.append(xts)
          tsd=pd.DataFrame(lits)
          return trd,tsd     

      def fit(self):
          dftr,dfts=self.pretreat()
          a=check_data(dftr,dfts,self.nc)
          file=open('Warning3.txt','w')
          file.write(str(a.fit())+' are not present in the training set'+"\n")
          file.write('Remove the samples containing these conditions and rerun')
          lt,lts=[],[]
          for i in range(2,self.nc+2):
              x_,y_=self.calculation(dftr,dfts,i,self.nc)
              li=pd.DataFrame(x_).transpose().add_suffix('_'+pd.DataFrame(dftr.iloc[:,i]).columns.values.tolist()[0])
              lits=pd.DataFrame(y_).transpose().add_suffix('_'+pd.DataFrame(dfts.iloc[:,i]).columns.values.tolist()[0])
              lt.append(li)
              lts.append(lits)
              ad=pd.concat(lt,axis=1, join='outer')
              adts=pd.concat(lts,axis=1, join='outer')
          adx=pd.concat([dftr.iloc[:,0:2].reset_index(),ad],axis=1,join='outer')
          adx=adx.drop(['index'],axis=1)
          adtsx=pd.concat([dfts.iloc[:,0:2].reset_index(),adts],axis=1,join='outer')
          adtsx=adtsx.drop(['index'],axis=1)
          return adx,adtsx