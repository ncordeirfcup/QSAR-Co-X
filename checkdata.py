import pandas as pd
class check_data():
      def __init__(self,dftr,dfts,nc):
            self.dftr=dftr
            self.dfts=dfts
            self.nc=nc
            
            
      def find_(self,df):
          li=[]
          n=self.nc+2
          c=df.iloc[:,2:n]
          ccv=c.columns
          for i in ccv:
              li.append(pd.unique(c[i]))
          return li

      def fit(self):
          trl=self.find_(self.dftr)
          tsl=self.find_(self.dfts)
          fltr=[item for sublist in trl for item in sublist]
          flts=[item for sublist in tsl for item in sublist]
          fltr,flts
          li2=[]
          for i in flts:
              if i not in fltr:
                 li2.append(i)
          return li2