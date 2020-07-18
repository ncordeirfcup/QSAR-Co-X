import pandas as pd
from checkdata import check_data
from boxjenk2 import boxjenk as bj
from sklearn.model_selection import train_test_split


class boxjenk():
      def __init__(self,df,dfts,nc):
            self.df=df
            self.dfts=dfts
            self.nc=nc
            
      def cal(self,cna,cnats,cond):
          dfi=cna.groupby([cond]).count()
          dfi['prob_'+str(cond)]=dfi[cna.iloc[:,0:1].columns]/cna.shape[0]
          x=pd.merge(cna,dfi,on=str(cond), how='left')
          x = x[x.columns.drop(list(x.filter(regex='_x')))]
          x = x[x.columns.drop(list(x.filter(regex='_y')))]
          x=x.drop([cond],axis=1)
          y=pd.merge(cnats,dfi,on=str(cond), how='left')
          y = y[y.columns.drop(list(y.filter(regex='_x')))]
          y = y[y.columns.drop(list(y.filter(regex='_y')))]
          y=y.drop([cond],axis=1)
          return x,y

      
      def cal2(self,dtr,trc,k):
          bs1=dtr.filter(regex=(k))
          bs2=trc.filter(regex=(k))
          l=[]
          for i in bs1:
              for j in bs2:
                  x=bs1[i]/bs2[j].values
                  l.append(x)
          #print(bs2)
          return l


      def ncal(self):
          self.dfts=self.dfts.reset_index()
          self.dfts=self.dfts.drop(['index'],axis=1)
          self.df=self.df.reset_index()
          self.df=self.df.drop(['index'],axis=1)
          cn=self.df.iloc[:,0:1]
          cc=self.df.iloc[:,2:(self.nc)+2]
          cna=pd.concat([cn,cc],axis=1)
          cnts=self.dfts.iloc[:,0:1]
          ccts=self.dfts.iloc[:,2:(self.nc)+2]
          cnats=pd.concat([cnts,ccts],axis=1)
          for i in range(self.nc):
               self.df=pd.concat([self.df,self.cal(cna,cnats,cc.columns[i])[0]],axis=1)
               self.dfts=pd.concat([self.dfts,self.cal(cna,cnats,cc.columns[i])[1]],axis=1)
               #self.dfts=self.dfts.drop(['index'],axis=1)
          tr=self.df
          ts=self.dfts
          
          
          #a=check_data(tr,ts,self.nc)
          #file=open('Warning3.txt','w')
          #tr.to_csv('trsave.csv', index=False)
          #ts.to_csv('tssave.csv', index=False)
          trc=tr.iloc[:,-self.nc:]
          tr2=tr.iloc[:,0:-self.nc]
          tsc=ts.iloc[:,-self.nc:]
          ts2=ts.iloc[:,0:-self.nc]
          bjk=bj(tr2,ts2,self.nc)
          dtr,dts=bjk.fit()
          dtr=dtr.sort_values(cn.columns[0])
          dtrs=dtr.iloc[:,0:2]
          dts=dts.sort_values(cn.columns[0])
          dtss=dts.iloc[:,0:2]
          #dtr.to_csv('dtr.csv', index=False)
          #dts.to_csv('dts.csv', index=False)
          xtr,xts=[],[]
          for k in cc.columns:
              yktr=pd.DataFrame(self.cal2(dtr,trc,k)).transpose().add_suffix('_p')
              ykts=pd.DataFrame(self.cal2(dts,tsc,k)).transpose().add_suffix('_p')
              save1=pd.concat([dtr,tsc], axis=1)
              #save1.to_csv('save1.csv',index=False)
              xtr.append(yktr)
              xts.append(ykts)
              fd=pd.concat(xtr,axis=1,join='outer')
              fdts=pd.concat(xts,axis=1,join='outer')
          fd2=pd.concat([dtrs,fd],axis=1,join='outer')
          fd2ts=pd.concat([dtss,fdts],axis=1,join='outer')
          return fd2,fd2ts
          
          