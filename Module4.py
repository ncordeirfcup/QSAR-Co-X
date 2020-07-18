import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk, Image
import os
import shutil
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames
import time
import pandas as pd
from sklearn.metrics import accuracy_score

initialdir=os.getcwd()


def data1():
    global filename1
    filename1 = askopenfilename(initialdir=initialdir,title = "Select Traning Set Result file")
    firstEntryTabFive.delete(0, END)
    firstEntryTabFive.insert(0, filename1)
    global c_
    c_,d_=os.path.splitext(filename1)
    global file1
    file1 = pd.read_csv(filename1)
    #global col3
    #col3 = list(file3.head(0))

def data2():
    global filename2
    filename2 = askopenfilename(initialdir=initialdir,title = "Open training set file")
    sixthEntryTabFive.delete(0, END)
    sixthEntryTabFive.insert(0, filename2)
    global file2
    file2 = pd.read_csv(filename2)
    #global col3
    #col3 = list(file3.head(0))
    
def data3():
    global filename3
    filename3 = askopenfilename(initialdir=initialdir,title = "Select Validation Set Result file")
    firstEntryTabFive_1.delete(0, END)
    firstEntryTabFive_1.insert(0, filename3)
    global g_
    g_,h_=os.path.splitext(filename3)
    global file3
    file3 = pd.read_csv(filename3)
    #global col3
    #col3 = list(file3.head(0))
    
def data4():
    global filename4
    filename4 = askopenfilename(initialdir=initialdir,title = "Open test set file")
    sixthEntryTabFive_2.delete(0, END)
    sixthEntryTabFive_2.insert(0, filename4)
    global file4
    file4 = pd.read_csv(filename4)
    #global col3
    #col3 = list(file3.head(0))
    
def Sol():
    file=file1.sort_values(by=file1.iloc[:,0:1].columns[0])
    #file2=file2.sort_values(by=file2.iloc[:,0:1].columns[0])
    df3=file[[file.iloc[:,0:1].columns[0],'Pred','Set']]
    nc=seventhEntryTabFive.get()
    df4=file2.iloc[:,0:(int(nc)+2)]
    df4=df4.sort_values(by=df4.iloc[:,0:1].columns[0])
    
    df5=pd.merge(df4,df3, on=df4.iloc[:,0:1].columns[0])
    #df5.to_csv('dfxx.csv', index=False)
    #df6=df5[df5['Set']=='Test']
    df6=df5[df5['Set']=='Test']
    #nc=seventhEntryTabFive.get()
    dfc=df6.iloc[:,2:(int(nc)+2)]
    dfc.columns.tolist()
    l=[]
    for name, group in df6.groupby(dfc.columns.tolist()):
        l.append(accuracy_score(group[df6.iloc[:,1:2].columns[0]], group['Pred'])*100)
    l1=pd.DataFrame(l).reset_index()
    l1.rename(columns={l1.iloc[:,1:2].columns[0]:'%Accuracy'}, inplace=True)
    #l1.columns=['%Accuracy']
    l2=pd.DataFrame(df6.groupby(dfc.columns.tolist()).size()).reset_index()
    l2.rename(columns={l2.iloc[:,-1:].columns[0]:'#Instances'}, inplace=True)
    l3=pd.concat([l2,l1], axis=1)
    l3=l3.drop(['index'],axis=1)
    #l3.set_index('index')
    l3.to_csv(str(c_)+'_cond'+'.csv', index=True)
    

def Sol2():
    file=file3.sort_values(by=file3.iloc[:,0:1].columns[0])
    #file2=file2.sort_values(by=file2.iloc[:,0:1].columns[0])
    df3=file[[file.iloc[:,0:1].columns[0],'Pred','Set']]
    #df3.to_csv('df3.csv')
    nc=seventhEntryTabFive_1.get()
    df4=file4.iloc[:,0:(int(nc)+2)]
    #df4=file2.iloc[:,0:(int(nc)+2)]
    df4=df4.sort_values(by=df4.iloc[:,0:1].columns[0])
    #df4.to_csv('df4.csv')
    df5=pd.merge(df4,df3, on=df4.iloc[:,0:1].columns[0])
    #df5.to_csv('dfxx.csv', index=False)
    #df6=df5[df5['Set']=='Test']
    #df6=df5[df5['Set']=='Test']
    #nc=seventhEntryTabFive.get()
    dfc=df5.iloc[:,2:(int(nc)+2)]
    dfc.columns.tolist()
    l=[]
    for name, group in df5.groupby(dfc.columns.tolist()):
        l.append(accuracy_score(group[df5.iloc[:,1:2].columns[0]], group['Pred'])*100)
    l1=pd.DataFrame(l).reset_index()
    #l1.to_csv('l5.csv', index=False)
    l1.rename(columns={l1.iloc[:,1:2].columns[0]:'%Accuracy'}, inplace=True)
    #l1.columns=['%Accuracy']
    l2=pd.DataFrame(df5.groupby(dfc.columns.tolist()).size()).reset_index()
    l2.rename(columns={l2.iloc[:,-1:].columns[0]:'#Instances'}, inplace=True)
    l3=pd.concat([l2,l1], axis=1)
    l3=l3.drop(['index'],axis=1)
    l3.to_csv(str(g_)+'_cond'+'.csv', index=False)

form = tk.Tk()
form.title("QSAR-Co-X (Module-4)")
form.geometry("650x350")

tab_parent = ttk.Notebook(form)

tab1 = ttk.Frame(tab_parent)

tab_parent.add(tab1, text="Condition-wise-prediction")
    
firstLabelTabFive_0_x = tk.Label(tab1, text="For test set",font=('Helvetica 12 bold'),anchor=W, justify=LEFT)
firstLabelTabFive_0_x.place(x=320, y=15)
firstLabelTabFive = tk.Label(tab1, text="Open training set result file",font=("Helvetica", 12),anchor=W, justify=LEFT)
firstLabelTabFive.place(x=60,y=40)
firstEntryTabFive = tk.Entry(tab1, width=40)
firstEntryTabFive.place(x=250,y=43)
b1=tk.Button(tab1,text='Browse', command=data1)
b1.place(x=500,y=40)
      
sixthLabelTabFive = tk.Label(tab1, text="Open training set file",font=('Helvetica 12'),anchor=W, justify=LEFT)
sixthLabelTabFive.place(x=90,y=70)
sixthEntryTabFive = tk.Entry(tab1, width=40)
sixthEntryTabFive.place(x=250,y=73)
b2=tk.Button(tab1,text='Browse', command=data2)
b2.place(x=500,y=70)

seventhLabelTabFive = tk.Label(tab1, text="Number of conditions",font=("Helvetica", 12),anchor=W, justify=LEFT)
seventhLabelTabFive.place(x=150,y=100)
seventhEntryTabFive = tk.Entry(tab1)
seventhEntryTabFive.place(x=310,y=103)
    
b3=Button(tab1, text='Submit', command=Sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b3.place(x=350,y=130)    


firstLabelTabFive_0 = tk.Label(tab1, text="For validation set",font=('Helvetica 12 bold'),anchor=W, justify=LEFT)
firstLabelTabFive_0.place(x=300, y=170)
firstLabelTabFive_1 = tk.Label(tab1, text="Open validation set result file",font=("Helvetica", 12),anchor=W, justify=LEFT)
firstLabelTabFive_1.place(x=35,y=190)
firstEntryTabFive_1 = tk.Entry(tab1, width=40)
firstEntryTabFive_1.place(x=250,y=193)
b4=tk.Button(tab1,text='Browse', command=data3)
b4.place(x=500,y=190)

sixthLabelTabFive_2 = tk.Label(tab1, text="Open test set file",font=('Helvetica 12'),anchor=W, justify=LEFT)
sixthLabelTabFive_2.place(x=90,y=220)
sixthEntryTabFive_2 = tk.Entry(tab1, width=40)
sixthEntryTabFive_2.place(x=250,y=223)
b5=tk.Button(tab1,text='Browse', command=data4)
b5.place(x=500,y=220)

seventhLabelTabFive_1 = tk.Label(tab1, text="Number of conditions",font=("Helvetica", 12),anchor=W, justify=LEFT)
seventhLabelTabFive_1.place(x=150,y=250)
seventhEntryTabFive_1 = tk.Entry(tab1)
seventhEntryTabFive_1.place(x=310,y=253)


b6=Button(tab1, text='Submit', command=Sol2,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b6.place(x=350,y=280)    

tab_parent.pack(expand=1, fill='both')

form.mainloop()