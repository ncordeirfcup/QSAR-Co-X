import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
import numpy as np
from testset_prediction import testset_prediction as tsp
from sklearn.feature_selection import VarianceThreshold

class stepwise_selection():
    def __init__(self,X, y,threshold_in,threshold_out,max_steps):
        self.X=X
        self.y=y
        self.threshold_in=threshold_in
        self.threshold_out=threshold_out
        self.max_steps=max_steps
        
    initial_list=[]
        
        
    def fit_linear_reg(self,X,y):
        x=np.ones(X.shape[0])
        x=list(x)
        x=pd.DataFrame(x)
        x.columns=['constant']
        X=pd.concat([X,x],axis=1)
        dp=pd.concat([X,y],axis=1)
        table=MANOVA.from_formula('X.values~ y.values', data=dp).mv_test().results['y.values']['stat']
        Wilks_lambda=table.iloc[0,0]
        F_value=table.iloc[0,3]
        p_value=table.iloc[0,4]
        return Wilks_lambda,F_value,p_value,table
    
  
    
    def feature_selection(self,X,y,verbose=True):
        #ideas were taken from
        #https://datascience.stackexchange.com/questions/24405/how-to-do-stepwise-regression-using-sklearn
        
        initial_list=[]
        included=list(initial_list)
        
        while True:
             changed=False
              # forward step
             excluded = list(set(X.columns)-set(included))
             new_pval = pd.Series(index=excluded)
             for new_column in excluded:
                 model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
                 new_pval[new_column] = model.pvalues[new_column]
             best_pval = new_pval.min()
              #model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[new_column]))).fit()
        
        
             if best_pval < self.threshold_in:
                try:
                   best_feature = new_pval.idxmin(axis=1)
                except ValueError:
                   best_feature = new_pval.idxmin()
                included.append(best_feature)
                #x_,y_,z_,t_=fit_linear_reg(X[included],y)
                changed=True
                if verbose:
                     print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))    
             # backward step
             model = sm.OLS(self.y, sm.add_constant(pd.DataFrame(self.X[included]))).fit()
             # use all coefs except intercept
             pvalues = model.pvalues.iloc[1:]
             worst_pval = pvalues.max() # null if pvalues is empty
             if worst_pval > self.threshold_out:
                changed=True
                worst_feature = pvalues.idxmax(axis=1)
                included.remove(worst_feature)
                if verbose:
                   print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
             if not changed or len(included)>=self.max_steps:
                break
        return included
        
    def fit_(self):
        #lda=LinearDiscriminantAnalysis()
        included_features=self.feature_selection(self.X,self.y)
        #lda.fit(self.X[included_features],self.y)
        table=self.fit_linear_reg(self.X[included_features],self.y)[3]
        wlambda=self.fit_linear_reg(self.X[included_features],self.y)[0]
        #accuracy=lda.score(self.X[included_features],self.y)
        fvalue=self.fit_linear_reg(self.X[included_features],self.y)[1]
        pvalue=self.fit_linear_reg(self.X[included_features],self.y)[2]
        #print('Selected features are: '+str(included_features))
        #print('Wilks lambda: '+str(wlambda))
        #print('F-value: '+str(fvalue))
        #print('p-value: '+str(pvalue))
        #ts=tsp(lda,self.X[included_features],self.y)
        #ts.fit()
        return included_features,wlambda,fvalue,pvalue