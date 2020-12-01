import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk, Image
#import pymysql
import os
import shutil
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames
import time
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from testset_prediction import testset_prediction as tsp
from cross_validation import cross_validation as cv
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot



initialdir=os.getcwd()
RF=RandomForestClassifier()

def data1():
    global filename1
    filename1 = askopenfilename(initialdir=initialdir,title = "Select sub-training file")
    firstEntryTabThree.delete(0, END)
    firstEntryTabThree.insert(0, filename1)
    global c_
    c_,d_=os.path.splitext(filename1)
    global file1
    file1 = pd.read_csv(filename1)
    global col1
    col1 = list(file1.head(0))
    
def data2():
    global filename2
    filename2 = askopenfilename(initialdir=initialdir,title = "Select test file")
    secondEntryTabThree.delete(0, END)
    secondEntryTabThree.insert(0, filename2)
    global file2
    file2 = pd.read_csv(filename2)
    
def data3():
    global filename3
    filename3 = askopenfilename(initialdir=initialdir,title = "Select validation file")
    fifthEntryTabOne.delete(0, END)
    fifthEntryTabOne.insert(0, filename3)
    global e_
    e_,f_=os.path.splitext(filename3)
    global file3
    file3 = pd.read_csv(filename3)

def data4():
    global filename4
    filename4 = askopenfilename(initialdir=initialdir,title = "Select parameter file")
    thirdEntryTabThree_x.delete(0, END)
    thirdEntryTabThree_x.insert(0, filename4)
    global e_x
    e_x,f_x=os.path.splitext(filename4)
    global file4
    file4 = pd.read_csv(filename4)
    
def correlation(X,cthreshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = X.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] > cthreshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in X.columns:
                    del X[colname] # deleting the column from the dataset
    return X   

def variance(X,threshold):
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=(threshold* (1 - threshold)))
    sel_var=sel.fit_transform(X)
    X=X[X.columns[sel.get_support(indices=True)]]    
    return X

def pretreat(X,cthreshold,vthreshold):
    X=correlation(X,cthreshold)
    X=variance(X,vthreshold)
    return X
    

def selected():
    x=file4.columns.to_list()
    ls,ls2=[],[]
    for i in file4.columns:
        ls.append((list(file4[i])))
    for i in ls:
        ls2.append([x for x in i if x!=0 and str(x)!='nan'])  
    param_grid = dict(zip(x,ls2))
    
    
    if Criterion.get()==1:
        estimator=RandomForestClassifier(random_state=4)
        rn='RF'
        param_grid['bootstrap']=[True,False]
        
    elif Criterion.get()==2:
        estimator=KNeighborsClassifier()
        rn='KNN'       
    elif Criterion.get()==3:
        estimator=BernoulliNB()
        rn='NB'
        param_grid['fit_prior']=[True,False] 
    elif Criterion.get()==4:
        estimator=SVC(probability=True)
        rn='SVC'        
    elif Criterion.get()==5:
         estimator=GradientBoostingClassifier(verbose=1,random_state=4)
         rn='GB'
         
             
    elif Criterion.get()==6:
         estimator=MLPClassifier()
         rn='MLP'
         if int(thirdEntryTabThreer3c1_h.get())!=0:
            param_grid ['hidden_layer_sizes']=[(int(thirdEntryTabThreer3c1_h.get()),)]
         if int(thirdEntryTabThreer3c1_h1.get())!=0:
            param_grid ['hidden_layer_sizes']=[(int(thirdEntryTabThreer3c1_h.get()),int(thirdEntryTabThreer3c1_h1.get()))]
         if int(thirdEntryTabThreer3c1_h2.get())!=0:
            param_grid ['hidden_layer_sizes']=[(int(thirdEntryTabThreer3c1_h.get()),int(thirdEntryTabThreer3c1_h1.get()),int(thirdEntryTabThreer3c1_h2.get()))]
         
    else:
        pass
    print(param_grid)
    rn='g_'+rn
    return estimator,param_grid,rn

   
 
def sol():
    Xtr=file1.iloc[:,2:]
    ytr=file1.iloc[:,1:2]
    ntr=file1.iloc[:,0:1]
    Xts=file2.iloc[:,2:]
    yts=file2.iloc[:,1:2]
    nts=file2.iloc[:,0:1]
    estimator,param_grid,rn=selected()
    cvg=thirdEntryTabOne.get()
    cvg=int(cvg)
    cvm=forthEntryTabOne.get()
    cvm=int(cvm)
    #param_grid=paramgrid()
    clf = GridSearchCV(cv=cvg, estimator=estimator, param_grid=param_grid, n_jobs=-1)
    cthreshold=float(thirdEntryTabThreer3c1.get())
    vthreshold=float(fourthEntryTabThreer5c1.get())
    Xtr=pretreat(Xtr,cthreshold,vthreshold)
    Xts=Xts[Xtr.columns]
    Xtr.to_csv('Pretreat_train_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    Xts.to_csv('Pretreat_test_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    clf.fit(Xtr,ytr)
    global clfb
    clfb=clf.best_estimator_
    clfb.fit(Xtr,ytr)
    #clf.best_estimator_.score(Xtr,ytr)
    #cvv=cv(Xtr,ytr,clf.best_estimator_,cvm)
    #print(clf.best_estimator_,clf.best_score_)
    #print(cvv.fit())
    #ts=tsp(Xts,yts,clf.best_estimator_)
    #print(ts.fit())
    #global filer
    
    filer = open(str(c_)+rn+"_tr.txt","w")

    filer.write('The best estimator is: '+'\n')
    filer.write(str(clf.best_estimator_))
    filer.write("\n")
    filer.write((str(cvm)+' fold cross validation statistics are: '+'\n'))
    filer.write("\n")
    writefile2(Xtr,ytr,clfb,cvm,filer)
    filer.write("\n")
    filer.write('The test set results are: '+'\n')
    filer.write("\n")
    a=writefile(Xts,yts,clfb,filer)
    filer.write("\n")
    b=pd.concat([file2,a],axis=1)
    b['Set'] = 'Test'
    b.to_csv(str(c_)+str(rn)+'_tspred.csv', index=False)
    label='test'
    color='blue'   
    global pyplot 
    pyplot.figure(figsize=(15,10))
    
    ROCplot(clfb,Xts,yts,label,color,pyplot)


def sol2():
    #file,plt=sol()
    estimator,pg,rn=selected()
    Xtr=file1.iloc[:,2:]
    cthreshold=float(thirdEntryTabThreer3c1.get())
    vthreshold=float(fourthEntryTabThreer5c1.get())
    Xtr=pretreat(Xtr,cthreshold,vthreshold)
    ytr=file1.iloc[:,1:2]
    ntr=file1.iloc[:,0:1]
    Xvd=file3[Xtr.columns]
    Xvd=Xvd[Xtr.columns]
    Xvd.to_csv('Pretreat_valid_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    
    if ytr.columns[0] in file3.columns:
        yvd=file3.iloc[:,1:2]
        nvd=file3.iloc[:,0:1]
        filer = open(str(c_)+rn+"_vd.txt","w")
        filer.write('The validation set results are: '+'\n')
        filer.write("\n")
        a=writefile(Xvd,yvd,clfb,filer)
        b=pd.concat([file3,a],axis=1)
        b['Set'] = 'Validation'
        b.to_csv(str(c_)+str(rn)+'_vdpred.csv', index=False) 
        label='validation'
        color='red'
        ROCplot(clfb,Xvd,yvd,label,color,pyplot)
    else:
        #est.fit(Xtr,ytr)
        yprvd=pd.DataFrame(clfb.predict(Xvd))
        yprvd.columns=['Pred']
        yprvd2=pd.DataFrame(clfb.predict_proba(Xvd))
        yprvd2.columns=['%Prob(-1)','%Prob(+1)']
        dfsvd=pd.concat([file3,yprvd2],axis=1)
        dfsvd['Diff']=abs(dfsvd['%Prob(-1)']-dfsvd['%Prob(+1)'])
        dfsvd['Outlier_info(Confidence estimation approach, Threshold 0.5)']=['In' if x>=0.5 else 'Outlier' for x in dfsvd['Diff']]
        dfsvd['Set'] = 'Screening'
        dfsvd.to_csv(str(c_)+str(rn)+'_scpred.csv', index=False)

def ROCplot(model,X,y,label,color,plt):
    #model.fit(X,y)
    s,p,t=selected()
    lr_probs =model.predict_proba(X)
    lr_probs = lr_probs[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(y, lr_probs)
    #pyplot.figure(figsize=(15,10))
    plt.plot(lr_fpr,lr_tpr, label=label, color=color, marker='.',  linewidth=1, markersize=10)
    plt.ylabel('True postive rate',fontsize=28)
    plt.xlabel('False postive rate',fontsize=28)
    plt.legend(fontsize=18)
    plt.tick_params(labelsize=18)
    rocn=str(c_)+str(t)+'_ROC.png'
    plt.savefig(rocn, dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, \
                      format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None,metadata=None)
             
    
def writefile(X,y,model,filerw):
    ts=tsp(X,y,model)
    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10=ts.fit()
    filerw.write('True Positive: '+str(a1)+"\n")
    filerw.write('True Negative: '+str(a2)+"\n")
    filerw.write('False Positive '+str(a3)+"\n")
    filerw.write('False Negative '+str(a4)+"\n")
    filerw.write('Sensitivity: '+str(a5)+"\n")
    filerw.write('Specificity: '+str(a6)+"\n")
    filerw.write('Accuracy: '+str(a7)+"\n")
    filerw.write('f1_score: '+str(a8)+"\n")
    #filer.write('Recall score: '+str(recall_score(self.y,ypred))
    filerw.write('MCC: '+str(a9)+"\n")
    filerw.write('ROC_AUC: '+str(a10)+"\n")   
    ypred=pd.DataFrame(model.predict(X))
    ypred.columns=['Pred']
    ypr2=pd.DataFrame(model.predict_proba(X))
    ypr2.columns=['%Prob(-1)','%Prob(+1)']
    dfsvd=pd.concat([ypred,ypr2],axis=1)
    dfsvd['Diff']=abs(dfsvd['%Prob(-1)']-dfsvd['%Prob(+1)'])
    dfsvd['Outlier_info(Confidence estimation approach, Threshold 0.5)']=['In' if x>=0.5 else 'Outlier' for x in dfsvd['Diff']]
    return dfsvd 

def writefile2(X,y,model,cvm,filerw):
    cvv=cv(X,y,model,cvm)
    a1,a2,a3,a4,a5,a6,a7,a8=cvv.fit()
    filerw.write('True Positive: '+str(a1)+"\n")
    filerw.write('True Negative: '+str(a2)+"\n")
    filerw.write('False Positive '+str(a3)+"\n")
    filerw.write('False Negative '+str(a4)+"\n")
    filerw.write('Sensitivity: '+str(a5)+"\n")
    filerw.write('Specificity: '+str(a6)+"\n")
    filerw.write('Accuracy: '+str(a7)+"\n")
    filerw.write('f1_score: '+str(a8)+"\n")
    #filer.write('Recall score: '+str(recall_score(self.y,ypred))
    #filerw.write('MCC: '+str(a9)+"\n")
    #filerw.write('ROC_AUC: '+str(a10)+"\n")  
    
form = tk.Tk()

form.title("QSAR-Co-X (Module-2)")

form.geometry("670x370")


tab_parent = ttk.Notebook(form)


tab1 = tk.Frame(tab_parent) #background='#ffffff')


tab_parent.add(tab1, text="Grid search based non-linear model")


###Tab1#####
    
firstLabelTabThree = tk.Label(tab1, text="Select sub-training set",font=("Helvetica", 12))
firstLabelTabThree.place(x=60,y=10)
firstEntryTabThree = tk.Entry(tab1, width=40)
firstEntryTabThree.place(x=230,y=13)
b3=tk.Button(tab1,text='Browse', command=data1,font=("Helvetica", 10))
b3.place(x=480,y=10)  

secondLabelTabThree = tk.Label(tab1, text="Select test set",font=("Helvetica", 12))
secondLabelTabThree.place(x=120,y=40)
secondEntryTabThree = tk.Entry(tab1,width=40)
secondEntryTabThree.place(x=230,y=43)
b4=tk.Button(tab1,text='Browse', command=data2,font=("Helvetica", 10))
b4.place(x=480,y=40)

thirdLabelTabThree_x = tk.Label(tab1, text="Select parameter file",font=("Helvetica", 12))
thirdLabelTabThree_x.place(x=70,y=70)
thirdEntryTabThree_x = tk.Entry(tab1,width=40)
thirdEntryTabThree_x.place(x=230,y=73)
b4_x=tk.Button(tab1,text='Browse', command=data4,font=("Helvetica", 10))
b4_x.place(x=480,y=70)

Criterion_Label = ttk.Label(tab1, text="Method:",font=("Helvetica", 12))
Criterion = IntVar()
#Criterion.set()
Criterion_RF = ttk.Radiobutton(tab1, text='Random Forest', variable=Criterion, value=1, command=selected)
Criterion_KNN = ttk.Radiobutton(tab1, text='k-nearest neighborhood', variable=Criterion, value=2, command=selected)
Criterion_NB = ttk.Radiobutton(tab1, text='BernoulliNB', variable=Criterion, value=3, command=selected)
Criterion_SVC = ttk.Radiobutton(tab1, text='Support Vector Classification', variable=Criterion, value=4, command=selected)
Criterion_GB = ttk.Radiobutton(tab1, text='Gradient boosting', variable=Criterion, value=5, command=selected)
Criterion_MLP = ttk.Radiobutton(tab1, text='Multilayer Perception', variable=Criterion, value=6, command=selected)


Criterion_Label.place(x=30,y=100)
Criterion_RF.place(x=100,y=100)
Criterion_KNN.place(x=210,y=100)
Criterion_NB.place(x=370,y=100)
Criterion_SVC.place(x=470, y=100)
Criterion_GB.place(x=65, y=130)
Criterion_MLP.place(x=195, y=130)

thirdLabelTabThreer2c1_h=Label(tab1, text='Hidden Layers (for MLP only)',font=("Helvetica", 10))
v1=IntVar(tab1,value=100)
v2=IntVar(tab1,value=0)
v3=IntVar(tab1,value=0)
thirdLabelTabThreer2c1_h.place(x=340,y=130)
thirdEntryTabThreer3c1_h=Entry(tab1,textvariable=v1,width=5)
thirdEntryTabThreer3c1_h.place(x=515,y=133)
thirdEntryTabThreer3c1_h1=Entry(tab1,textvariable=v2,width=5)
thirdEntryTabThreer3c1_h1.place(x=565,y=133)
thirdEntryTabThreer3c1_h2=Entry(tab1,textvariable=v3,width=5)
thirdEntryTabThreer3c1_h2.place(x=615, y=133)

thirdLabelTabThreer2c1=Label(tab1, text='Correlation cut-off',font=("Helvetica", 12))
thirdLabelTabThreer2c1.place(x=150,y=160)
thirdEntryTabThreer3c1=Entry(tab1)
thirdEntryTabThreer3c1.place(x=300,y=160)

fourthLabelTabThreer4c1=Label(tab1, text='Variance cut-off',font=("Helvetica", 12))
fourthLabelTabThreer4c1.place(x=150,y=190)
fourthEntryTabThreer5c1=Entry(tab1)
fourthEntryTabThreer5c1.place(x=300,y=190)

thirdLabelTabOne=tk.Label(tab1, text="CV for grid search",font=("Helvetica", 12))
thirdLabelTabOne.place(x=150,y=220)
thirdEntryTabOne = tk.Entry(tab1, width=20)
thirdEntryTabOne.place(x=300,y=220)

forthLabelTabOne=tk.Label(tab1, text="CV for model predictability",font=("Helvetica", 12))
forthLabelTabOne.place(x=100,y=250)
forthEntryTabOne = tk.Entry(tab1, width=20)
forthEntryTabOne.place(x=300,y=250)
#forthLabelTabOne=tk.Label(tab1, text="parameter grid",font=("Helvetica", 12))
#forthLabelTabOne.place(x=120,y=185)
#v = StringVar()
#forthEntryTabOne = tk.Entry(tab1, textvariable=v, width=40)
#forthEntryTabOne.place(x=230,y=185)

b2=Button(tab1, text='Generate model', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b2.place(x=450,y=205)

fifthLabelTabOne = tk.Label(tab1, text="Select validation set",font=("Helvetica", 12))
fifthLabelTabOne.place(x=60,y=280)
fifthEntryTabOne = tk.Entry(tab1, width=40)
fifthEntryTabOne.place(x=230,y=283)
b3=tk.Button(tab1,text='Browse', command=data3,font=("Helvetica", 10))
b3.place(x=480,y=280)  

b4=Button(tab1, text='Submit', command=sol2,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b4.place(x=300,y=305)

tab_parent.pack(expand=1, fill='both')


form.mainloop()