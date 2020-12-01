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
        estimator=RandomForestClassifier(random_state=3)
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
         estimator=GradientBoostingClassifier(verbose=1,random_state=3)
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
def enable():
    E10T1['state']='normal'
    
def disable():
    E10T1['state']='disabled'

def enable1():
    E6T1['state']='normal'
    
def disable1():
    E6T1['state']='disabled'    

form = tk.Tk()

form.title("QSAR-Co-X (Module-2)")

form.geometry("670x370")

tab_parent = ttk.Notebook(form)

tab1 = tk.Frame(tab_parent, background='#ffffff')

tab_parent.add(tab1, text="Grid search based non-linear model")

####TAB2#####
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

thirdLabelTabThreer2c1=Label(tab1, text='Correlation cut-off',font=("Helvetica", 12))
thirdLabelTabThreer2c1.place(x=45,y=70)
thirdEntryTabThreer3c1=Entry(tab1, width=7)
thirdEntryTabThreer3c1.place(x=180,y=73)


fourthLabelTabThreer4c1=Label(tab1, text='Variance cut-off',font=("Helvetica", 12))
fourthLabelTabThreer4c1.place(x=250,y=70)
fourthEntryTabThreer5c1=Entry(tab1, width=7)
fourthEntryTabThreer5c1.place(x=370,y=73)

forthLabelTabOne=tk.Label(tab1, text="CV(model predictability)",font=("Helvetica", 12))
forthLabelTabOne.place(x=450,y=70)
forthEntryTabOne = tk.Entry(tab1, width=7)
forthEntryTabOne.place(x=630,y=73)

var1= IntVar()
CB1=Checkbutton(tab1, text = "KNN",  variable=var1, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
CB1.place(x=60,y=100)

L1T1= tk.Label(tab1, text="n_neighbors",font=("Helvetica", 10))
L1T1.place(x=25,y=130)
E1T1 = tk.Entry(tab1, width=10)
E1T1.place(x=103,y=133)

Criterion2 = StringVar()
Criterion3 = StringVar()

L2T1= tk.Label(tab1, text="Weights",font=("Helvetica", 10))
L2T1.place(x=25,y=160)
C1= ttk.Radiobutton(tab1, text='Uniform', variable=Criterion2, value='uniform')
C2= ttk.Radiobutton(tab1, text='Distance', variable=Criterion2,  value='distance')
C1.place(x=80,y=160)
C2.place(x=160,y=160)

L3T1= tk.Label(tab1, text="Algorithm",font=("Helvetica", 10))
L3T1.place(x=20,y=190)
C3= ttk.Radiobutton(tab1, text='Auto', variable=Criterion3, value='auto')
C4= ttk.Radiobutton(tab1, text='Ball_tree', variable=Criterion3, value='ball_tree')
C5= ttk.Radiobutton(tab1, text='KD_tree', variable=Criterion3, value='kd_tree')
C6= ttk.Radiobutton(tab1, text='Brute', variable=Criterion3, value='brute')

C3.place(x=80,y=190)
C4.place(x=130,y=190)
C5.place(x=80,y=210)
C6.place(x=150,y=210)

b5=Button(tab1, text='Generate', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b5.place(x=100,y=250)

#############################

var2= IntVar()
CB2=Checkbutton(tab1, text = "Bernoulli Naive Bayes",  variable=var2, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
CB2.place(x=450,y=100)

L4T1= tk.Label(tab1, text="alpha",font=("Helvetica", 10))
L4T1.place(x=455,y=130)
E2T1 = tk.Entry(tab1, width=10)
E2T1.place(x=503,y=133)

Criterion4 = BooleanVar()
#Criterion4.set(True)
L5T1= tk.Label(tab1, text="Fit_prior",font=("Helvetica", 10))
L5T1.place(x=450,y=160)
C7= ttk.Radiobutton(tab1, text='True', variable=Criterion4, value=True)
C8= ttk.Radiobutton(tab1, text='False', variable=Criterion4,  value=False)
C7.place(x=510,y=160)
C8.place(x=570,y=160)

b6=Button(tab1, text='Generate', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b6.place(x=500,y=190)

########################################

var3= IntVar()
CB2=Checkbutton(tab1, text = "Support Vector Classification",  variable=var3, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
CB2.place(x=200,y=100)

L6T1= tk.Label(tab1, text="C",font=("Helvetica", 10))
L6T1.place(x=280,y=130)
E3T1 = tk.Entry(tab1, width=10)
E3T1.place(x=303,y=133)

L7T1= tk.Label(tab1, text="Gamma",font=("Helvetica", 10))
L7T1.place(x=250,y=160)
E4T1 = tk.Entry(tab1, width=10)
E4T1.place(x=303,y=160)

Criterion5 = StringVar()

L8T1= tk.Label(tab1, text="Kernel",font=("Helvetica", 10))
L8T1.place(x=250,y=190)
C9= ttk.Radiobutton(tab1, text='Linear', variable=Criterion5, value='linear')
C10= ttk.Radiobutton(tab1, text='RBF', variable=Criterion5, value='rbf')
C9_x= ttk.Radiobutton(tab1, text='Poly', variable=Criterion5, value='poly')
C10_x= ttk.Radiobutton(tab1, text='Sigmoid', variable=Criterion5, value='sigmoid')
C9.place(x=290,y=190)
C10.place(x=370,y=190)
C9_x.place(x=290,y=210)
C10_x.place(x=370,y=210)

b6=Button(tab1, text='Generate', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b6.place(x=315,y=235)

#################

var4= IntVar()
CB3=Checkbutton(tab1, text = "Multilayer Perception",  variable=var4, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
CB3.place(x=30,y=280)

L9T1= tk.Label(tab1, text="alpha",font=("Helvetica", 10))
L9T1.place(x=60,y=310)
E5T1 = tk.Entry(tab1, width=10)
E5T1.place(x=100,y=313)

thirdLabelTabThreer2c1_h=Label(tab1, text='Hidden Layers',font=("Helvetica", 10))
v1=IntVar(tab1,value=100)
v2=IntVar(tab1,value=0)
v3=IntVar(tab1,value=0)
thirdLabelTabThreer2c1_h.place(x=10,y=340)
thirdEntryTabThreer3c1_h=Entry(tab1,textvariable=v1,width=5)
thirdEntryTabThreer3c1_h.place(x=110,y=343)
thirdEntryTabThreer3c1_h1=Entry(tab1,textvariable=v2,width=5)
thirdEntryTabThreer3c1_h1.place(x=150,y=343)
thirdEntryTabThreer3c1_h2=Entry(tab1,textvariable=v3,width=5)
thirdEntryTabThreer3c1_h2.place(x=190, y=343)

Criterion6 = StringVar()
Criterion7 = StringVar()
Criterion8 = StringVar()


L8T1= tk.Label(tab1, text="Activation",font=("Helvetica", 10))
L8T1.place(x=30,y=370)
C11= ttk.Radiobutton(tab1, text='Identity', variable=Criterion6, value='identity')
C12= ttk.Radiobutton(tab1, text='Logistic', variable=Criterion6, value='logistic')
C13= ttk.Radiobutton(tab1, text='Tanh', variable=Criterion6, value='tanh')
C14= ttk.Radiobutton(tab1, text='Relu', variable=Criterion6, value='relu')

C11.place(x=90,y=370)
C12.place(x=160,y=370)
C13.place(x=90,y=390)
C14.place(x=160,y=390)

L9T1= tk.Label(tab1, text="Solver",font=("Helvetica", 10))
L9T1.place(x=30,y=420)
C15= ttk.Radiobutton(tab1, text='SGD', variable=Criterion7, value='sgd')
C16= ttk.Radiobutton(tab1, text='Adam', variable=Criterion7, value='adam')
C15.place(x=90,y=420)
C16.place(x=160,y=420)

L10T1= tk.Label(tab1, text="Learning rate",font=("Helvetica", 10))
L10T1.place(x=10,y=460)
C17= ttk.Radiobutton(tab1, text='Constant', variable=Criterion8, value='constant')
C18= ttk.Radiobutton(tab1, text='Adaptive', variable=Criterion8, value='adaptive')
C19= ttk.Radiobutton(tab1, text='Invscaling', variable=Criterion8, value='invscaling')

C17.place(x=90,y=460)
C18.place(x=160,y=460)
C19.place(x=90,y=480)

b6=Button(tab1, text='Generate', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b6.place(x=100,y=510)


###################
var5= IntVar()
CB3=Checkbutton(tab1, text = "Random Forest",  variable=var5, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
CB3.place(x=280,y=280)

L11T1= tk.Label(tab1, text="Max depth",font=("Helvetica", 10))
L11T1.place(x=240,y=310)
Criterion25=StringVar()
MaxFeatures_None = ttk.Radiobutton(tab1, text='None', variable=Criterion25, value='none',command=disable1)   
MaxFeatures_None.place(x=310,y=313)
MaxFeatures_integer = ttk.Radiobutton(tab1, text='Number', variable=Criterion25, value='integer',command=enable1)   
MaxFeatures_integer.place(x=370,y=313)                                          
E6T1 = tk.Entry(tab1,textvariable=MaxFeatures_integer, width=7,state=DISABLED)
E6T1.place(x=380,y=333)
Criterion25.set('none')

L12T1= tk.Label(tab1, text="Minimum samples leaf",font=("Helvetica", 10))
L12T1.place(x=240,y=360)
E7T1 = tk.Entry(tab1, width=10)
E7T1.place(x=380,y=363)


L13T1= tk.Label(tab1, text="Minimum samples split",font=("Helvetica", 10))
L13T1.place(x=240,y=390)
E8T1 = tk.Entry(tab1, width=10)
E8T1.place(x=380,y=393)

L14T1= tk.Label(tab1, text="Number of Estimators",font=("Helvetica", 10))
L14T1.place(x=240,y=420)
E9T1 = tk.Entry(tab1, width=10)
E9T1.place(x=380,y=423)


Criterion9 = StringVar()
Criterion10 = StringVar()
Criterion11 = BooleanVar()


L15T1= tk.Label(tab1, text="Criterion",font=("Helvetica", 10))
L15T1.place(x=240,y=480)
C20= ttk.Radiobutton(tab1, text='Gini', variable=Criterion9, value='gini')
C21= ttk.Radiobutton(tab1, text='Entropy', variable=Criterion9, value='entropy')

C20.place(x=310,y=480)
C21.place(x=360,y=480)


L9T1= tk.Label(tab1, text="Max features",font=("Helvetica", 10))
L9T1.place(x=240,y=450)
C15= ttk.Radiobutton(tab1, text='Auto', variable=Criterion10, value='auto')
C16= ttk.Radiobutton(tab1, text='SQRT', variable=Criterion10, value='sqrt')
C16_1= ttk.Radiobutton(tab1, text='Log2', variable=Criterion10, value='log2')
C15.place(x=320,y=450)
C16.place(x=370,y=450)
C16_1.place(x=420,y=450)

L10T1= tk.Label(tab1, text="Bootstrap",font=("Helvetica", 10))
L10T1.place(x=240,y=510)
C17= ttk.Radiobutton(tab1, text='True', variable=Criterion11, value=True)
C18= ttk.Radiobutton(tab1, text='False', variable=Criterion11, value=False)

C17.place(x=310,y=510)
C18.place(x=360,y=510)


b6=Button(tab1, text='Generate', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b6.place(x=330,y=530)

###################

var6= IntVar()
CB3=Checkbutton(tab1, text = "Gradient Boosting",  variable=var6, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
CB3.place(x=500,y=220)

L11T1= tk.Label(tab1, text="Max depth",font=("Helvetica", 10))
L11T1.place(x=460,y=250)
Criterion26=StringVar()
Criterion26.set('none')
MaxFeatures1_None = ttk.Radiobutton(tab1, text='None', variable=Criterion26, value='none',command=disable)   
MaxFeatures1_None.place(x=530,y=250)
MaxFeatures1_integer = ttk.Radiobutton(tab1, text='Number', variable=Criterion26, value='integer',command=enable)   
MaxFeatures1_integer.place(x=610,y=250)                                          

E10T1 = tk.Entry(tab1, textvariable=MaxFeatures1_integer, width=7,state=DISABLED)
E10T1.place(x=620,y=270)

L12T1= tk.Label(tab1, text="Minimum samples leaf",font=("Helvetica", 10))
L12T1.place(x=470,y=300)
E11T1 = tk.Entry(tab1, width=10)
E11T1.place(x=610,y=303)


L13T1= tk.Label(tab1, text="Minimum samples split",font=("Helvetica", 10))
L13T1.place(x=470,y=330)
E12T1 = tk.Entry(tab1, width=10)
E12T1.place(x=610,y=333)

L14T1= tk.Label(tab1, text="Number of Estimators",font=("Helvetica", 10))
L14T1.place(x=470,y=360)
E13T1 = tk.Entry(tab1, width=10)
E13T1.place(x=610,y=363)

L14T1= tk.Label(tab1, text="Learning rate",font=("Helvetica", 10))
L14T1.place(x=520,y=390)
E14T1 = tk.Entry(tab1, width=10)
E14T1.place(x=610,y=390)

L14T1= tk.Label(tab1, text="Subsample",font=("Helvetica", 10))
L14T1.place(x=530,y=420)
E15T1 = tk.Entry(tab1, width=10)
E15T1.place(x=610,y=420)

Criterion22 = StringVar()
Criterion23 = StringVar()
Criterion24 = StringVar()


L15T1= tk.Label(tab1, text="Criterion",font=("Helvetica", 10))
L15T1.place(x=480,y=450)
C20= ttk.Radiobutton(tab1, text='Friedman MSE', variable=Criterion22, value='friedman_mse')
C21= ttk.Radiobutton(tab1, text='MAE', variable=Criterion22, value='mae')

C20.place(x=540,y=450)
C21.place(x=640,y=450)


L16T1= tk.Label(tab1, text="Loss",font=("Helvetica", 10))
L16T1.place(x=480,y=475)
C22= ttk.Radiobutton(tab1, text='Deviance', variable=Criterion23, value='deviance')
C23= ttk.Radiobutton(tab1, text='Exponential', variable=Criterion23, value='exponential')
C22.place(x=520,y=475)
C23.place(x=600,y=475)

L17T1= tk.Label(tab1, text="Max features",font=("Helvetica", 10))
L17T1.place(x=460,y=500)
C24= ttk.Radiobutton(tab1, text='Log2', variable=Criterion24, value='log2')
C25= ttk.Radiobutton(tab1, text='SQRT', variable=Criterion24, value='sqrt')
C26= ttk.Radiobutton(tab1, text='Auto', variable=Criterion24, value='auto')

C24.place(x=540,y=500)
C25.place(x=590,y=500)
C26.place(x=640,y=500)

b7=Button(tab1, text='Generate', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b7.place(x=560,y=530)


fifthLabelTabOne = tk.Label(tab1, text="Select validation set",font=("Helvetica", 12))
fifthLabelTabOne.place(x=60,y=560)
fifthEntryTabOne = tk.Entry(tab1, width=40)
fifthEntryTabOne.place(x=230,y=563)
b3=tk.Button(tab1,text='Browse', command=data3,font=("Helvetica", 10))
b3.place(x=480,y=560)  

b4=Button(tab1, text='Submit', command=sol2,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b4.place(x=320,y=590)

tab_parent.pack(expand=1, fill='both')

form.mainloop()
