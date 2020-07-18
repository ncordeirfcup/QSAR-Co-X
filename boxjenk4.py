import pandas as pd
from checkdata import check_data
#from boxjenk import boxjenk as bj   #use this for method 4 mentioned in the manual
from boxjenk2 import boxjenk as bj  #use this for method 5 mentioned in the manual
from sklearn.model_selection import train_test_split


class boxjenk():
      def __init__(self,df,dfts,nc):
            self.df=df
            self.dfts=dfts
            self.nc=nc
            

           
      def cal2(self,dtr,trc,k):
          bs1=dtr.filter(regex=(k))
          bs2=trc.filter(regex=(k))
          l=[]
          for i in bs1:
              for j in bs2:
                  x=bs1[i]/bs2[j].values
                  l.append(x)
          return l


      def ncal(self):
          cn=self.df.iloc[:,0:1]
          ca=self.df.iloc[:,1:2]
          cc=self.df.iloc[:,2:(self.nc)+2]
          cd=self.df.iloc[:,(self.nc)+2:]
          cna=pd.concat([cn,cc],axis=1)

          tr=self.df
          ts=self.dfts
          tr=tr.sort_values(cn.columns[0])
          ts=ts.sort_values(cn.columns[0])
          
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

      
