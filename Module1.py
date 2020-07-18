import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk, Image
#import pymysql
import os
import shutil
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames
import time
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from stepwise_selection import stepwise_selection as ss
from sequential_selection import stepwise_selection as sq
from testset_prediction import testset_prediction as tsp
#from .file import stepwise_selection as ss
#from .file import testset_prediction as tsp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import sys
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from ycr1 import ycrandom
from boxjenk import boxjenk as bj
from boxjenk2 import boxjenk as bj2
from boxjenk3 import boxjenk as bj3
from boxjenk4 import boxjenk as bj4
from applicability import apdom
from sklearn.model_selection import train_test_split
from kmca import kmca



#import db_config
initialdir=os.getcwd()
model=LinearDiscriminantAnalysis()
#initialdir='C:\\Users\\Amit\\Downloads\\pyqsar_tutorial-master\\Best_model'
#initialdir='C:\\Users\\USER\\Downloads\\twitter_classification_project'
def data():
    filename = askopenfilename(initialdir=initialdir,title = "Select file")
    firstEntryTabOne.delete(0, END)
    firstEntryTabOne.insert(0, filename)
    global a_
    a_,b_=os.path.splitext(filename)
    global file
    file = pd.read_csv(filename)
    
    #file=sys.argv[1]

    #global col
    #col = list(file.head(0))
    #print(col)
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
    #global col2
    #col2 = list(file2.head(0))
    
def data3():
    global filename3
    filename3 = askopenfilename(initialdir=initialdir,title = "Select validation file")
    thirdEntryTabFive.delete(0, END)
    thirdEntryTabFive.insert(0, filename3)
    global e_
    e_,f_=os.path.splitext(filename3)
    global file3
    file3 = pd.read_csv(filename3)
    #global col3
    #col3 = list(file3.head(0))
    
def data4():
    global filename4
    filename4 = askopenfilename(initialdir=initialdir,title = "Select Traning Set Result file")
    firstEntryTabFive.delete(0, END)
    firstEntryTabFive.insert(0, filename4)
    global file4
    file4 = pd.read_csv(filename4)
    #global col3
    #col3 = list(file3.head(0))
    
def data5():
    global filename5
    filename5 = askopenfilename(initialdir=initialdir,title = "Open training set file")
    sixthEntryTabFive.delete(0, END)
    sixthEntryTabFive.insert(0, filename5)
    global g_
    g_,h_=os.path.splitext(filename5)
    global file7
    file7 = pd.read_csv(filename5)
    #global col3
    #col3 = list(file3.head(0))

   
def sol1(): 
    if Selection.get()=='Predefined':
       setn=file.iloc[:,-1:].columns[0]
       tr=file[file[setn]=='Train']
       ts=file[file[setn]=='Test']
       tr=tr.drop(setn,axis=1)
       ts=ts.drop(setn,axis=1)
    elif Selection.get()=='Random':
       perc=secondEntryTabOne.get()
       perc=float(perc)/100
       seed=thirdEntryTabOne.get()
       a,b= train_test_split(file,test_size=perc, random_state=int(seed))
       tr=pd.DataFrame(a)
       ts=pd.DataFrame(b)
    elif Selection.get()=='KMCA':
       perc=secondEntryTabOne.get()
       perc=float(perc)/100
       nclus=thirdEntryTabOne_x.get()
       nc=firstEntryTabTwo.get()
       seed=thirdEntryTabOne.get()
       kmc=kmca(file,int(nc),perc,int(seed),int(nclus))
       tr,ts=kmc.cal()
       
    return tr,ts

def solsave():
    tr,ts=sol1()
    #savename1 = filedialog.asksaveasfilename(initialdir=initialdir,title = "Save trainingset file")
    #savename1= savename1.split('.')[0] + '_tr.csv'
    savename1= str(a_) + '_tr.csv'
    tr.to_csv(savename1,index=False)
    #savename2 = filedialog.asksaveasfilename(initialdir=initialdir,title = "Save testset file")
    savename2= str(a_) + '_ts.csv'
    ts.to_csv(savename2,index=False)
    #print(a_,savename1,savename2)
   
def sol2():
    nc=firstEntryTabTwo.get()
    nc=int(nc)
    a,b=sol1()
    
    if var3.get():
       bjk=bj(a,b,nc)
       i,j=bjk.fit()
    elif var4.get():
       bjk=bj2(a,b,nc)
       i,j=bjk.fit()
    elif var5.get():
       bjk=bj3(a,b,nc)
       i,j=bjk.ncal()   
    elif var6.get():
       #c=pd.concat([a,b],axis=0)
       bjk=bj4(a,b,nc)
       i,j=bjk.ncal()
    #i.to_csv('train_s2.csv',index=False)
    #j.to_csv('vdsetbj.csv',index=False)
    return i,j
    
def sol3():
    from sklearn.model_selection import train_test_split
    tr,vd=sol2()
    perc=secondEntryTabTwo.get()
    #print(perc)
    perc=float(perc)/100
    seed=thirdEntryTabTwo.get()
    st,ts= train_test_split(tr,test_size=perc, random_state=int(seed))

    return st,ts,vd
    
def solsave2():
    st,ts,vd=sol3()
    #savename1 = filedialog.asksaveasfilename(initialdir=initialdir,title = "Save Sub-trainingset")
    savename1= str(a_) + '_strbj.csv'
    st.to_csv(savename1,index=False)
    #savename2 = filedialog.asksaveasfilename(initialdir=initialdir,title = "Save testset")
    savename2= str(a_) + '_tsbj.csv'
    ts.to_csv(savename2,index=False)
    #savename3 = filedialog.asksaveasfilename(initialdir=initialdir,title = "Save validationset")
    savename3= str(a_) + '_vdbj.csv'
    vd.to_csv(savename3,index=False)
    
  
    
def trainsetfit(X,y):
    corrl=thirdEntryTabThreer3c1.get()
    var=fourthEntryTabThreer5c1.get()
    ms=fifthBoxTabThreer6c1.get()
    ti=sixthEntryTabThree.get()
    to=seventhEntryTabThree.get()
    sl=ss(X,y,float(corrl),float(var),float(ti),float(to),int(ms))
    a,b,c,d=sl.fit_()
    pt=sl.pretreat(X)
    #pt.to_csv('Pretreat.csv')
    pt.to_csv('pt_train_'+str(corrl)+'_'+str(var)+'.csv')
    return a,b,c,d

def trainsetfit2(X,y):
    cthreshold=thirdEntryTabThreer3c2.get()
    vthreshold=fourthEntryTabThreer5c2.get()
    max_steps=fifthBoxTabThreer6c2.get()
    flot=Criterion.get()
    forw=Criterion3.get()
    score=Criterion4.get()
    cvl=fifthBoxTabThreer7c2.get()
    sqs=sq(X,y,float(cthreshold),float(vthreshold),int(max_steps),flot,forw,score,int(cvl))
    #sqs=sq(dfX, dfy,0.95,0.01,4,True,True,'accuracy',0)
    a,b,c,d=sqs.fit_()
    pt=sqs.pretreat(X)
    pt.to_csv('pt_train_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    return a,b,c,d
    #print(float(cthreshold),float(vthreshold),int(max_steps),flot,forw,score,int(cvl))
   


def ROCplot(X,y):
    model.fit(X,y)
    lr_probs =model.predict_proba(X)
    lr_probs = lr_probs[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(y, lr_probs)
    return lr_fpr, lr_tpr

def writefile1():
    Xtr=file1.iloc[:,2:]
    ytr=file1.iloc[:,1:2]
    ntr=file1.iloc[:,0:1]
    Xts=file2.iloc[:,2:]
    yts=file2.iloc[:,1:2]
    nts=file2.iloc[:,0:1]
    if var1.get():
       a,b,c,d=trainsetfit(Xtr,ytr)
       filer = open(str(c_)+"_fslda.txt","w")
    elif var2.get():
       a,b,c,d=trainsetfit2(Xtr,ytr)
       filer = open(str(c_)+"_sfslda.txt","w")
    
    #filer = open("resultsx.txt","w")
    filer.write("Sub-training set results "+"\n")
    filer.write("\n")
    #file3.write("Selected features are:"+str(a)+"\n")
    filer.write("Wilks lambda: "+str(b)+"\n")
    filer.write("Fvalue: "+str(c)+"\n")
    filer.write("pvalue: "+str(d)+"\n")
    model.fit(Xtr[a],ytr)
    filer.write("Selected features :"+str(a)+"\n")
    filer.write("intercept: "+str(model.intercept_)+"\n")
    filer.write("coefficients: "+str(model.coef_)+"\n")
    yprtr=pd.DataFrame(model.predict(Xtr[a]))
    yprtr.columns=['Pred']
    yprtr2=pd.DataFrame(model.predict_proba(Xtr[a]))
    yprtr2.columns=['%Prob(-1)','%Prob(+1)']
    adstr=apdom(Xtr[a],Xtr[a])
    yadstr=adstr.fit()
    dfstr=pd.concat([ntr,Xtr[a],ytr,yprtr,yprtr2,yadstr],axis=1)
    dfstr['Set'] = 'Sub_train'
    yprts=pd.DataFrame(model.predict(Xts[a]))
    yprts.columns=['Pred']
    yprts2=pd.DataFrame(model.predict_proba(Xts[a]))
    yprts2.columns=['%Prob(-1)','%Prob(+1)']
    adts=apdom(Xts[a],Xtr[a])
    yadts=adts.fit()
    dfsts=pd.concat([nts,Xts[a],yts,yprts,yprts2,yadts],axis=1)
    dfsts['Set'] = 'Test'
    tb=Xtr[a].corr()
    tbn=str(c_)+'_corr.csv'
    tb.to_csv(tbn)
    
    

    finda=pd.concat([dfstr,dfsts],axis=0)
    #finda.to_csv('find.csv',index=False)
    #savename4 = filedialog.asksaveasfilename(initialdir=initialdir,title = "Save File with Predicted Activity")
    savename4= str(c_) + '_pred.csv'
    finda.to_csv(savename4,index=False)
    writefile2(Xtr[a],ytr,model,filer)
    filer.write("\n")
    filer.write("Test set results: "+"\n")
    filer.write("\n")
    writefile2(Xts[a],yts,model,filer)
    
    filer.close()
    
def writefile2(X,y,model,filerw):
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
    
def writefile3():   
    nf=secondEntryTabFive.get()
    nf=int(nf)
    global file5
    file5=file4[file4['Set']=='Sub_train']
    #global Xtr5
    Xtr5=file5.iloc[:,1:nf+1]
    ytr5=file5.iloc[:,nf+1:nf+2]
    file6=file4[file4['Set']=='Test']
    Xts=file6.iloc[:,1:nf+1]
    yts=file6.iloc[:,nf+1:nf+2]
    model.fit(Xtr5,ytr5)
    Xvd=file3[Xtr5.columns]
    if ytr5.columns[0] in file3.columns: 
       yvd=file3[ytr5.columns]
       nvd=file3.iloc[:,0:1]
       yprvd=pd.DataFrame(model.predict(Xvd))
       yprvd.columns=['Pred']
       yprvd2=pd.DataFrame(model.predict_proba(Xvd))
       yprvd2.columns=['%Prob(-1)','%Prob(+1)']
       advd=apdom(Xvd,Xtr5)
       yadvd=advd.fit()
       dfsvd=pd.concat([nvd,Xvd,yvd,yprvd,yprvd2,yadvd],axis=1)
       dfsvd['Set'] = 'Validation'
       #findv=pd.concat([dfstr,dfsts],axis=0)
       #finda.to_csv('find.csv',index=False)
       #savename4 = filedialog.asksaveasfilename(initialdir=initialdir,title = "Save File with Predicted Activity")
       savename4= str(e_) + '_pred.csv'
       dfsvd.to_csv(savename4,index=False)
       filer2 = open(str(e_)+"_pred.txt","w")
       filer2.write("Validation set results: "+"\n")
       filer2.write("\n")
       writefile2(Xvd,yvd,model,filer2)
       e,f=ROCplot(Xtr5,ytr5)
       g,h=ROCplot(Xts,yts)
       i,j=ROCplot(Xvd,yvd)
       pyplot.figure(figsize=(15,10))
       pyplot.plot(e,f, label='Sub-train', color='blue', marker='.',  linewidth=1, markersize=10)
       pyplot.plot(g,h, label='Test', color='red', marker='.', linewidth=1, markersize=10)
       pyplot.plot(i,j, label='Validation', color='green', marker='.', linewidth=1, markersize=10)
       pyplot.ylabel('True postive rate',fontsize=28)
       pyplot.xlabel('False postive rate',fontsize=28)
       pyplot.legend(fontsize=18)
       pyplot.tick_params(labelsize=18)
       rocn=str(e_)+'_ROC.png'
       pyplot.savefig(rocn, dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, \
                      format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None,metadata=None)
        
    else:
       #filer2 = open("resultvd.txt","w")
       #vd=pd.DataFrame(np.zeros(Xvd.shape[0]))
       nvd=file3.iloc[:,0:1]
       yprvd=pd.DataFrame(model.predict(Xvd))
       yprvd.columns=['Pred']
       yprvd2=pd.DataFrame(model.predict_proba(Xvd))
       yprvd2.columns=['%Prob(-1)','%Prob(+1)']
       advd=apdom(Xvd,Xtr5)
       yadvd=advd.fit()
       dfsvd=pd.concat([nvd,Xvd,yprvd,yprvd2,yadvd],axis=1)
       dfsvd['Set'] = 'Screening'
       #findv=pd.concat([dfstr,dfsts],axis=0)
       #finda.to_csv('find.csv',index=False)
       #savename4 = filedialog.asksaveasfilename(initialdir=initialdir,title = "Save File with Predicted Activity")
       savename4= str(e_) + '_scpred.csv'
       dfsvd.to_csv(savename4,index=False)
       e,f=ROCplot(Xtr5,ytr5)
       g,h=ROCplot(Xts,yts)
       pyplot.figure(figsize=(15,10))
       pyplot.plot(e,f, label='Sub-train', color='blue', marker='.',  linewidth=1, markersize=10)
       pyplot.plot(g,h, label='Test', color='red', marker='.', linewidth=1, markersize=10)
       pyplot.ylabel('True postive rate',fontsize=28)
       pyplot.xlabel('False postive rate',fontsize=28)
       pyplot.legend(fontsize=18)
       pyplot.tick_params(labelsize=18)
       rocn=str(e_)+'_ROC.png'
       pyplot.savefig(rocn, dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, \
                      format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None,metadata=None)
    
def ycrand(): 
    s=file4[[file4.iloc[:,0:1].columns[0],file4.iloc[:,-1:].columns[0]]]
    df=pd.merge(file7,s, on=file4.iloc[:,0:1].columns[0])
    #dft=df[df['Set']=='Sub_train']
    nd=secondEntryTabFive.get()
    nd=int(nd)
    nc=seventhEntryTabFive.get()
    nc=int(nc)
    #nc=len(desc)
    Xtr5=file4.iloc[:,1:nd+1]
    desc=Xtr5.columns
    ni=ninthEntryTabFive.get()
    ni=int(ni)
    
    if var7.get():
       a=ycrandom(df,nc,desc,ni,model,1)    
    elif var8.get():
       a=ycrandom(df,nc,desc,ni,model,2)
    elif var9.get():
       a=ycrandom(df,nc,desc,ni,model,3)
    elif var10.get():
       a=ycrandom(df,nc,desc,ni,model,4)
    filer3 = open(str(g_)+"_ycresult.txt","w")
    filer3.write('The yc-randomized Wilks lambda value after '+str(ni)+' run is '+str(a.randomization()[0])+"\n")
    filer3.write('The yc-randomized accuracy after '+str(ni)+' run is '+str(a.randomization()[1]))
    
   
    
def enable():
    secondLabelTabOne['state']='normal'
    secondEntryTabOne['state']='normal'
    thirdLabelTabOne['state']='normal'
    thirdEntryTabOne['state']='normal'
    thirdLabelTabOne_x['state']='disabled'
    thirdEntryTabOne_x['state']='disabled'   

def disable():
    secondLabelTabOne['state']='disabled'
    secondEntryTabOne['state']='disabled'
    thirdLabelTabOne['state']='disabled'
    thirdEntryTabOne['state']='disabled' 
    

def enable2():
    secondLabelTabOne['state']='normal'
    secondEntryTabOne['state']='normal'
    thirdLabelTabOne['state']='normal'
    thirdEntryTabOne['state']='normal'
    thirdLabelTabOne_x['state']='normal'
    thirdEntryTabOne_x['state']='normal'


form = tk.Tk()
form.title("QSAR-Co-X (Module-1)")
form.geometry("650x350")

tab_parent = ttk.Notebook(form)

tab1 = ttk.Frame(tab_parent)
tab2 = ttk.Frame(tab_parent)
tab3 = ttk.Frame(tab_parent)
tab4 = ttk.Frame(tab_parent)
#tab5 = ttk.Frame(tab_parent)



tab_parent.add(tab1, text="Data preparation")
tab_parent.add(tab2, text="Linear model development")
tab_parent.add(tab3, text="Screening/validation")


# === WIDGETS FOR TAB ONE
#first
firstLabelTabOne = tk.Label(tab1, text="Select Data",font=("Helvetica", 12))
firstLabelTabOne.place(x=90,y=10)
firstEntryTabOne = tk.Entry(tab1,text='',width=50)
firstEntryTabOne.place(x=180,y=13)
b5=tk.Button(tab1,text='Browse', command=data,font=("Helvetica", 10))
b5.place(x=500,y=10)

firstLabelTabTwo_1=tk.Label(tab1, text='Number of conditions',font=("Helvetica", 12))
firstLabelTabTwo_1.place(x=150,y=40)
firstEntryTabTwo = tk.Entry(tab1)
firstEntryTabTwo.place(x=310,y=40)


secondLabelTabOne_1=Label(tab1, text='Dataset division techniques',font=('Helvetica 12 bold'))
secondLabelTabOne_1.place(x=220,y=60)

Selection = StringVar()
Selection.set('Predefined')
Criterion_sel1 = ttk.Radiobutton(tab1, text='Predefined', variable=Selection, value='Predefined',command=disable)
Criterion_sel2 = ttk.Radiobutton(tab1, text='Random Division', variable=Selection, value='Random',command=enable)
Criterion_sel3 = ttk.Radiobutton(tab1, text='KMCA', variable=Selection, value='KMCA',command=enable2)
Criterion_sel1.place(x=50,y=80)
Criterion_sel2.place(x=250,y=80)
Criterion_sel3.place(x=500,y=80)


secondLabelTabOne=Label(tab1, text='%Data-points(validation set)',font=("Helvetica", 12), justify='center',state=DISABLED)
secondLabelTabOne.place(x=80,y=100)
secondEntryTabOne=Entry(tab1, state=DISABLED)
secondEntryTabOne.place(x=290,y=105)
thirdLabelTabOne=Label(tab1, text='Seed value',font=("Helvetica", 12), state=DISABLED)
thirdLabelTabOne.place(x=200,y=135)
thirdEntryTabOne=Entry(tab1, state=DISABLED)
thirdEntryTabOne.place(x=290,y=135)


thirdLabelTabOne_x=Label(tab1, text='Number of clusters',font=("Helvetica", 12), state=DISABLED)
thirdLabelTabOne_x.place(x=480,y=105)
thirdEntryTabOne_x=Entry(tab1, state=DISABLED)
thirdEntryTabOne_x.place(x=480,y=135)


b6=tk.Button(tab1, text='Generate train-test sets', bg="orange", command=solsave,font=("Helvetica", 10))
b6.place(x=280,y=160)

firstLabelTabTwo = tk.Label(tab1, text="Box Jenkins based moving average",font=('Helvetica 12 bold'))
firstLabelTabTwo.place(x=200,y=190)

var3= IntVar()
N1 = Checkbutton(tab1, text = "Method-1",  variable=var3, \
                 font=("Helvetica", 12))
N1.place(x=150, y=210)

var4= IntVar()
N1 = Checkbutton(tab1, text = "Method-2",  variable=var4, \
                 font=("Helvetica", 12))
N1.place(x=250, y=210)

var5= IntVar()
N1 = Checkbutton(tab1, text = "Method-3",  variable=var5, \
                 font=("Helvetica", 12))
N1.place(x=350, y=210)

var6= IntVar()
N1 = Checkbutton(tab1, text = "Method-4",  variable=var6, \
                 font=("Helvetica", 12))
N1.place(x=450, y=210)

secondLabelTabTwo=Label(tab1, text='%Data-points(test set)',font=("Helvetica", 12))
secondLabelTabTwo.place(x=140,y=240)
secondEntryTabTwo=Entry(tab1)
secondEntryTabTwo.place(x=310,y=240)

thirdLabelTabTwo=Label(tab1, text='Seed value',font=("Helvetica", 12))
thirdLabelTabTwo.place(x=220,y=265)
thirdEntryTabTwo=Entry(tab1)
thirdEntryTabTwo.place(x=310,y=265)

b7=tk.Button(tab1, text='Generate subtrain-test-validation sets', bg="orange", command=solsave2,font=("Helvetica", 10))
b7.place(x=220,y=290)
#################################

# === WIDGETS FOR TAB Three
firstLabelTabThree = tk.Label(tab2, text="Select sub-training set",font=("Helvetica", 12))
firstLabelTabThree.place(x=60,y=10)
firstEntryTabThree = tk.Entry(tab2, width=40)
firstEntryTabThree.place(x=230,y=13)
b3=tk.Button(tab2,text='Browse', command=data1,font=("Helvetica", 10))
b3.place(x=480,y=10)

secondLabelTabThree = tk.Label(tab2, text="Select test set",font=("Helvetica", 12))
secondLabelTabThree.place(x=120,y=40)
secondEntryTabThree = tk.Entry(tab2,width=40)
secondEntryTabThree.place(x=230,y=43)
b4=tk.Button(tab2,text='Browse', command=data2,font=("Helvetica", 10))
b4.place(x=480,y=40)
#thirdLabelTabThreer1c1=Label(tab3, text='FS-LDA model',font=("Helvetica", 12),anchor=W, justify=LEFT).grid(row=3, column=2)

var1= IntVar()
C1 = Checkbutton(tab2, text = "FS-LDA",  variable=var1, \
                 font=("Helvetica", 12))
C1.place(x=200, y=85)

#thirdLabelTabThreer2c1=Label(tab2, text='Correlation cutoff',font=("Helvetica", 12)).grid(row=4, column=1)
thirdLabelTabThreer2c1=Label(tab2, text='Correlation cutoff',font=("Helvetica", 12))
thirdLabelTabThreer2c1.place(x=35,y=110)
thirdEntryTabThreer3c1=Entry(tab2)
thirdEntryTabThreer3c1.place(x=165,y=110)

#fourthLabelTabThreer4c1=Label(tab2, text='Variance cutoff',font=("Helvetica", 12)).grid(row=5, column=1)
fourthLabelTabThreer4c1=Label(tab2, text='Variance cutoff',font=("Helvetica", 12))
fourthLabelTabThreer4c1.place(x=40,y=135)
fourthEntryTabThreer5c1=Entry(tab2)
fourthEntryTabThreer5c1.place(x=165,y=135)

fifthLabelTabThreer6c1 = Label(tab2, text= 'Maximum steps',font=("Helvetica", 12))
fifthLabelTabThreer6c1.place(x=45,y=160)
fifthBoxTabThreer6c1= Spinbox(tab2, from_=0, to=100, width=5)
fifthBoxTabThreer6c1.place(x=165,y=160)

sixthLabelTabThree=Label(tab2, text= 'p-value to enter',font=("Helvetica", 12))
sixthLabelTabThree.place(x=35,y=185)
sixthEntryTabThree=Entry(tab2)
sixthEntryTabThree.place(x=165,y=185)

#seventhLabelTabThree=Label(tab2,text='p-value to remove',font=("Helvetica", 12)).grid(row=8,column=1,sticky=E)
seventhLabelTabThree=Label(tab2,text='p-value to remove',font=("Helvetica", 12))
seventhLabelTabThree.place(x=35,y=210)
seventhEntryTabThree=Entry(tab2)
seventhEntryTabThree.place(x=165,y=210)

b1=Button(tab2, text='Generate model', command=writefile1,bg="orange",font=("Helvetica", 10))
b1.place(x=170, y=240)

#thirdLabelTabThreer1c2=Label(tab3, text='SFS-LDA model',font=("Helvetica", 12),anchor=W, justify=LEFT).grid(row=3, column=4)
var2= IntVar()
C2 = Checkbutton(tab2, text = "SFS-LDA",  variable=var2, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
#C2.grid(row=3, column=4)
C2.place(x=400,y=85)

thirdLabelTabThreer2c2=Label(tab2, text='Correlation cutoff',font=("Helvetica", 12))
thirdLabelTabThreer2c2.place(x=350,y=110)
thirdEntryTabThreer3c2=Entry(tab2)
thirdEntryTabThreer3c2.place(x=475,y=110)

fourthLabelTabThreer4c2=Label(tab2, text='Variance cutoff',font=("Helvetica", 12))
fourthLabelTabThreer4c2.place(x=350,y=135)
fourthEntryTabThreer5c2=Entry(tab2)
fourthEntryTabThreer5c2.place(x=475,y=135)

fifthLabelTabThreer6c2 = Label(tab2, text= 'Maximum steps',font=("Helvetica", 12))
fifthLabelTabThreer6c2.place(x=350,y=160)
fifthBoxTabThreer6c2= Spinbox(tab2, from_=0, to=100, width=5)
fifthBoxTabThreer6c2.place(x=475,y=160)


fifthLabelTabThreer7c2 = Label(tab2, text= 'Cross_validation',font=("Helvetica", 12))
fifthLabelTabThreer7c2.place(x=350,y=185)
fifthBoxTabThreer7c2= Spinbox(tab2, from_=0, to=100, width=5)
fifthBoxTabThreer7c2.place(x=475,y=185)

Criterion_Label = ttk.Label(tab2, text="Floating:",font=("Helvetica", 12))
Criterion = BooleanVar()
Criterion.set(False)
Criterion_Gini = ttk.Radiobutton(tab2, text='True', variable=Criterion, value=True)
Criterion_Entropy = ttk.Radiobutton(tab2, text='False', variable=Criterion, value=False)
Criterion_Label.place(x=350,y=210)
Criterion_Gini.place(x=420,y=210)
Criterion_Entropy.place(x=470,y=210)

Criterion_Label3 = ttk.Label(tab2, text="Forward:",font=("Helvetica", 12))
Criterion3 = BooleanVar()
Criterion3.set(True)
Criterion_Gini2 = ttk.Radiobutton(tab2, text='True', variable=Criterion3, value=True)
#Criterion_Gini2.pack(column=4, row=9, sticky=(W))
Criterion_Entropy2 = ttk.Radiobutton(tab2, text='False', variable=Criterion3, value=False)
Criterion_Label3.place(x=350,y=235)
Criterion_Gini2.place(x=420,y=235)
Criterion_Entropy2.place(x=470,y=235)


Criterion_Label4 = ttk.Label(tab2, text="Scoring:",font=("Helvetica", 12),anchor=W, justify=LEFT)
Criterion4 = StringVar()
Criterion4.set('accuracy')
Criterion_acc3 = ttk.Radiobutton(tab2, text='Accuracy', variable=Criterion4, value='accuracy')
#Criterion_prec3 = ttk.Radiobutton(tab3, text='Precision', variable=Criterion4, value='precision')
Criterion_roc3 = ttk.Radiobutton(tab2, text='ROC_AUC', variable=Criterion4, value='roc_auc')
Criterion_Label4.place(x=350,y=260)
Criterion_acc3.place(x=420,y=260)
#Criterion_prec3.grid(column=4, row=10, sticky=(N))
Criterion_roc3.place(x=490,y=260)

b2=Button(tab2, text='Generate model', command=writefile1,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b2.place(x=420,y=285)


# === WIDGETS FOR TAB Four

firstLabelTabFive = tk.Label(tab3, text="Open training set result file",font=("Helvetica", 12),anchor=W, justify=LEFT)
firstLabelTabFive.place(x=60,y=10)
firstEntryTabFive = tk.Entry(tab3, width=40)
firstEntryTabFive.place(x=250,y=13)
b8=tk.Button(tab3,text='Browse', command=data4)
b8.place(x=500,y=10)

secondLabelTabFive=Label(tab3,text='Number of descriptors',font=("Helvetica", 12),anchor=W, justify=LEFT)
secondLabelTabFive.place(x=65,y=45)
secondEntryTabFive=Entry(tab3)
secondEntryTabFive.place(x=310,y=50)

thirdLabelTabFive = tk.Label(tab3, text="Select validation/screening set",font=("Helvetica", 12),anchor=W, justify=LEFT)
thirdLabelTabFive.place(x=30,y=75)
thirdEntryTabFive = tk.Entry(tab3, width=40)
thirdEntryTabFive.place(x=250,y=80)
b9=tk.Button(tab3,text='Browse', command=data3)
b9.place(x=500,y=75)

b10=Button(tab3, text='Submit', command=writefile3,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b10.place(x=350,y=115)

forthLabelTabFive = tk.Label(tab3, text="Yc randomization",font=('Helvetica 12 bold'),anchor=W, justify=CENTER)
forthLabelTabFive_x = tk.Label(tab3, text="(Import training set result + number of descriptors from above)",font=('Helvetica 9 bold'),anchor=W, justify=CENTER)
forthLabelTabFive.place(x=70,y=150)
forthLabelTabFive_x.place(x=210,y=152)


var7= IntVar()
N1 = Checkbutton(tab3, text = "Method-1",  variable=var7, \
                 font=("Helvetica", 12))
N1.place(x=150, y=170)

var8= IntVar()
N1 = Checkbutton(tab3, text = "Method-2",  variable=var8, \
                 font=("Helvetica", 12))
N1.place(x=250, y=170)

var9= IntVar()
N1 = Checkbutton(tab3, text = "Method-3",  variable=var9, \
                 font=("Helvetica", 12))
N1.place(x=350, y=170)

var10= IntVar()
N1 = Checkbutton(tab3, text = "Method-4",  variable=var10, \
                 font=("Helvetica", 12))
N1.place(x=450, y=170)


seventhLabelTabFive = tk.Label(tab3, text="Number of conditions",font=("Helvetica", 12),anchor=W, justify=LEFT)
seventhLabelTabFive.place(x=150,y=230)
seventhEntryTabFive = tk.Entry(tab3)
seventhEntryTabFive.place(x=310,y=230)


sixthLabelTabFive = tk.Label(tab3, text="Open training set file",font=('Helvetica 12'),anchor=W, justify=LEFT)
sixthLabelTabFive.place(x=90,y=200)
sixthEntryTabFive = tk.Entry(tab3, width=40)
sixthEntryTabFive.place(x=250,y=203)
b11=tk.Button(tab3,text='Browse', command=data5)
b11.place(x=500,y=200)

#eighthLabelTabFive=Label(tab3,text='Number of independent variables',font=("Helvetica", 12),anchor=W, justify=LEFT)
#eighthLabelTabFive.place(x=65,y=250)
#eighthEntryTabFive=Entry(tab3)
#eighthEntryTabFive.place(x=310,y=250)

ninthLabelTabFive=Label(tab3,text='Number of runs',font=("Helvetica", 12),anchor=W, justify=LEFT)
ninthLabelTabFive.place(x=195,y=260)
ninthEntryTabFive=Entry(tab3)
ninthEntryTabFive.place(x=310,y=260)

b10=Button(tab3, text='Submit', command=ycrand,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b10.place(x=350,y=290)

tab_parent.pack(expand=1, fill='both')

form.mainloop()