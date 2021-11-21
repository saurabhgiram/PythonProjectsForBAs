#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:

class project:
    
keys = {1 : 'Age', 2: 'LoS', 3: 'noofinvestigation', 4: 'nooftreatment',5: 'noofpatients', 6: 'HRG', 7 : 'DayofWeek'}
hrg_keys = {1:'VB02Z', 2: 'VB03Z', 3: 'VB04Z', 4: 'VB05Z', 5: 'VB06Z', 6: 'VB07Z', 7: 'VB08Z', 8: 'VB09Z', 9: 'VB11Z'}
dow_keys = {1: 'Friday', 2: 'Monday', 3: 'Saturday', 4: 'Sunday', 5: 'Thursday', 6: 'Tuesday', 7:'Wednesday' }


# In[3]:


def single_ifs(x):
    for i in range(1,len(keys)+1):
        if(x == i):
            z = keys[i]
    return z


# In[4]:



def ifs(x,y):
    for i in range(1,len(keys)+1):
        if(x == i):
            z = keys[i]
            for j in range(1,len(keys)+1):
                if( y == j):
                    p = keys[j]
    print(z,p)
    return z,p


# In[5]:


def hrg_values(v):
    for i in range(1,len(hrg_keys)+1):
        if(x == i):
            z = keys[i]
    return z


# In[6]:


def dow_values(z):
    
    for i in range(1,len(dow_keys)+1):
        if(x == i):
            z = keys[i]
    return z  


# In[7]:


dfsample = pd.read_csv("AEDsample.csv")
dfsample # also print type of data


# In[8]:


def trying(choice):
    if (choice == 'Age'):
        ch = input("Which Variable do you want to plot Age with \n 2: LoS   3: noofinvestigation  4: nooftreatments  5:noofpatients  6: HRG  7: DayofWeek \n")
    
    elif (choice == 'LoS'):
        ch = input("Which Variable do you want to plot LoS with \n 1: Age 3: noofinvestigation 4: nooftreatments 5: noofpatients 6: HRG  7: DayofWeek \n")
    
    elif (choice == 'noofinvestigation'):
        ch = input("Which Variable do you want to plot noofinvestigation with \n 1: Age 2: LoS 4: nooftreatments 5: noofpatients 6: HRG  7: DayofWeek \n")

    elif (choice == 'noofpatients'):
        ch = input("Which Variable do you want to plot noofpatients with \n 1: Age 2: LoS 3: noofinvestigation 5: noofpatients 6: HRG  7: DayofWeek \n")
    
    elif (choice == 'noofpatients'):
        ch = input("Which Variable do you want to plot noofpatients with \n 1: Age 2: LoS 3: noofinvestigation 4: nooftreatments 6: HRG  7: DayofWeek \n")
    
    elif (choice == 'HRG'):
        ch = input("Which Variable do you want to plot HRG with \n 1: Age 2: LoS 3: noofinvestigation 4: nooftreatments 5: noofpatients 7: DayofWeek \n")
    
    elif (choice == 'DayofWeek'):
        ch = input("Which Variable do you want to plot HRG with \n 1: Age 2: LoS 3: noofinvestigation 4: nooftreatments 5: noofpatients 6: HRG \n")

    return int(ch)


# In[9]:


sample = dfsample.sample(n = 400, random_state = 530)  # n = sample size , random_state sets the seed


# In[10]:


sample.columns # columns within the sample dataframe


# In[11]:


sample.describe()


# In[12]:


A = pd.DataFrame(sample.describe().drop(['count','25%','50%','75%']))
A


# In[13]:


type(A.iat[2,0])


# In[14]:


def patient_info():
    
    Patient_Id = input(" Please provide the id of patient ? ")
    Patient_info = sample.loc[sample['ID'] == Patient_Id]
    if Patient_info.empty == True:
        print("Oops! That was no valid number. Try again...")
    else:
        print(Patient_info)


# In[15]:


def correlation(a,b,data = sample):
    corr =  round(data[a].corr(data[b]) , 3)
    return corr


# In[16]:


def full_correlation(data = sample):
    corr = sample.corr()
    ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');


# In[17]:


def variable_range(c):

    if(c == "Age"):
         inp4, inp5 = input("Choose between "+A.iloc[2,0].astype(str)+" and "+A.iloc[3,0].astype(str)+"\n").split()
    elif(c == "LoS"):
         inp4, inp5 = input("Choose between "+A.iloc[2,3].astype(str)+" and "+A.iloc[3,3].astype(str)+"\n").split()
    elif(c == "noofinvestigation"):
         inp4, inp5 = input("Choose between "+A.iloc[2,4].astype(str)+" and "+A.iloc[3,4].astype(str)+"\n").split()
    elif(c == "nooftreatment"):
         inp4, inp5 = input("Choose between "+A.iloc[2,5].astype(str)+" and "+A.iloc[3,5].astype(str)+"\n").split()       
    elif(c == "noofpatients"):
         inp4, inp5 = input("Choose between "+A.iloc[2,6].astype(str)+" and "+A.iloc[3,6].astype(str)+"\n").split()

    range_data = sample[(sample[c] >= int(inp4) ) & (sample[c] <= int(inp5))]
    return range_data


# In[18]:


def categorical(c,data = sample):
    if(c == "Age"):
        sg = int(input("Choose one between "+A.iloc[2,0].astype(str)+" and "+A.iloc[3,0].astype(str)+"\n"))
    
    elif(c == "LoS"):
        sg = int(input("Choose one between "+A.iloc[2,3].astype(str)+" and "+A.iloc[3,3].astype(str)+"\n"))
    
    elif(c == "noofinvestigation"):
        sg = int(input("Choose one between "+A.iloc[2,4].astype(str)+" and "+A.iloc[3,4].astype(str)+"\n"))
    
    elif(c == "nooftreatment"):
        sg = int(input("Choose one between "+A.iloc[2,5].astype(str)+" and "+A.iloc[3,5].astype(str)+"\n"))
    
    elif(c == "noofpatients"):
        sg = int(input("Choose one between "+A.iloc[2,6].astype(str)+" and "+A.iloc[3,6].astype(str)+"\n"))
    
    elif(c == "HRG"):
        column_values = data[["HRG"]].values
        unique_values = np.unique(column_values)
        print("Choose one from: ")
        for i in range(len(unique_values)):
            print(str(i+1)+" : "+ unique_values[i])
        inp = int(input())
        sg = hrg_values(inp)
   
    elif(c == "DayofWeek"):
        column_values = data[["DayofWeek"]].values
        unique_values = np.unique(column_values)
        for i in range(len(unique_values)):
            print(str(i+1)+" : "+ unique_values[i])
        inp = int(input())
        sg = dow_values(inp) 
        
    unique_data = data.loc[data[c] == sg]
    return unique_data


# In[31]:


def plots(f,j,data = sample):

    if(j==1):  #HISTOGRAM
        if(f=='Age'):
            if(f == "Age"):
                inputAge = int(input("Length of age range of interest:"))
                graphChild = Graph("Age","Frequency",inputAge)
                graphChild.PlotHistogram(graphChild)
            
        elif(f == 'LoS'):
            q = int(input("Input size of buckets for Length of Stay:):"))

            bins_list = np.arange(0, 500+q+1, q).tolist()

            n, bins, patches = plt.hist(x=data["LoS"], bins= bins_list, color='OliveDrab',
                            alpha=0.65, rwidth=0.85)

            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('LoS')
            plt.ylabel('Frequency')
            plt.title('Histogram - LoS Minutes')
            maxfreq = n.max()
            # Set a clean upper y-axis limit.
            plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
            
        elif(f == 'noofinvestigation'):
            
            q = int(input("Length of noofpatients  range of interest:"))
            bins_list = np.arange(0, 69, q).tolist()

            n, bins, patches = plt.hist(x=data["noofinvestigation"], bins= bins_list, color='Orange',
                            alpha=0.65, rwidth=0.85)

            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('noofinvestigation')
            plt.ylabel('Frequency')
            plt.title('Histogram - noofinvestigation')
            maxfreq = n.max()
            # Set a clean upper y-axis limit.
            plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
            
        elif(f == 'nooftreatment'):
            print("Error")
        elif(f == 'noofpatients'):
            data["noofpatients"].plot.hist(bins = 20, figsize=(7,4), color="#27B7CB", rwidth=0.9, title="Number of Patients")
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('noofpatients')
            plt.ylabel('Frequency')
            plt.title('Histogram - noofpatients')
        elif(f == 'HRG'):
            print("Error")
        elif(f == 'DayofWeek'):
            print("Error")
    elif(j==2): # Box Plot
        
        if(f=='Age'):
            data["Age"].plot.box() 
        elif(f == 'LoS'):
            data["LoS"].plot.box()
        elif(f == 'noofinvestigation'):
            data["noofinvestigation"].plot.box()
        elif(f == 'nooftreatment'):
            data["nooftreatment"].plot.box()
        elif(f == 'noofpatients'):
            data["noofpatients"].plot.box()
        
        
    elif(j==3): # PIE CHART
        if(f=='Age'):
            print("Error")
        elif(f == 'LoS'):
            print("Error")
        elif(f == 'noofinvestigation'):
            print("Error")
        elif(f == 'nooftreatment'):
            print("Error")
        elif(f == 'noofpatients'):
            print("Error")
        elif(f == 'HRG'):
            print("Error")
        elif(f == 'DayofWeek'):
            print("Error")
               
        
    elif(j==4): # Dist Plot - A Distplot or distribution plot, depicts the variation in the data distribution
        if(f=='Age'):
            print("Error")
        elif(f == 'LoS'):
            print("Error")
        elif(f == 'noofinvestigation'):
            print("Error")
        elif(f == 'nooftreatment'):
            print("Error")
        elif(f == 'noofpatients'):
            sns.distplot(data["noofpatients"], color="g", bins=100)
            
        elif(f == 'HRG'):
            print("Error")
        elif(f == 'DayofWeek'):
            print("Error")


# In[ ]:


#def plots_recommendation():
    
    


# In[28]:


def plots_simple(f,g,h,data= sample):
    if(h==1): # SCATTER PLOTS
        
        if(f == 'Age' or g =='Age'):
            if(f == 'LoS' or g == 'LoS'):
                print("Error")
                
            
            elif(f == 'noofinvestigation' or g == 'noofinvestigation'):
                plt.scatter(data["noofinvestigation"], data["Age"], color = "#FF51FF")
                plt.title("Scatter plot of Age VS No of Investigation")
                plt.ylabel("Age")
                plt.xlabel("No of Investigations")
                
            elif(f == 'nooftreatment' or g == 'nooftreatment'):
                plt.scatter(data[f],data[g])
            
            elif(f == 'noofpatients' or g == 'noofpatients'):
                plt.scatter(data[f],data[g])
            
            elif(f == 'HRG' or g == 'HRG'):
                plt.scatter(data["HRG"], data["Age"])
                plt.title("Scatter plot of HRG VS Age")
                plt.ylabel("Age")
                plt.xlabel("HRG")
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                plt.scatter(data[f],data[g])
                
                
        
        elif(f == 'LoS' or g =='LoS'):
            if(f == 'noofinvestigation' or g == 'noofinvestigation'):
                plt.scatter(data["noofinvestigation"], data["LoS"], color = "#FF51FF")
                plt.title("Scatter plot of LoS VS No of Investigation")
                plt.ylabel("LoS")
                plt.xlabel("No of Investigations")
                
            elif(f == 'nooftreatment' or g == 'nooftreatment'):
                print("Error")
            
            elif(f == 'noofpatients' or g == 'noofpatients'):
                a = list(range(0,69))
                plt.bar(data["noofpatients"], dfsample["LoS"])
                plt.title("LoS by noofpatients")
                plt.ylabel("Length of Stay")
                plt.xlabel("noofpatients")
            
            elif(f == 'HRG' or g == 'HRG'):
                plt.scatter(data["HRG"], data["LoS"])
                plt.title("Scatter plot of HRG VS Length of stay")
                plt.ylabel("Length of stay ")
                plt.xlabel("HRG")
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                plt.scatter(data[f],data[g])
                
        
        elif(f == 'noofinvestigation' or g =='noofinvestigation'):
            if(f == 'nooftreatment' or g == 'nooftreatment'):
                plt.scatter(data["noofinvestigation"], data["nooftreatment"], color = "#FF51FF")
                plt.title("Scatter plot of  VS No of Investigation")
                plt.ylabel("No. of treatment")
                plt.xlabel("No of Investigations")
            
            elif(f == 'noofpatients' or g == 'noofpatients'):
                plt.scatter(data["noofinvestigation"], data["noofpatients"], color = "#FF51FF")
                plt.title("Scatter plot of Number of Patients VS No of Investigation")
                plt.ylabel("No. of patients")
                plt.xlabel("No of Investigations")
            
            elif(f == 'HRG' or g == 'HRG'):
                plt.scatter(data["HRG"], data["noofinvestigation"], color = "#FF51FF")
                plt.title("Scatter plot of HRG VS Number of investigations")
                plt.ylabel("Number of investigations")
                plt.xlabel("HRG")
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                plt.scatter(data["noofinvestigation"], data["DayofWeek"], color = "#FF51FF")
                plt.title("Scatter plot of Day of Week VS No of Investigation")
                plt.ylabel("Day of Week")
                plt.xlabel("No of Investigations")
                
        elif(f == 'nooftreatment' or g =='nooftreatment'):
            
            if(f == 'noofpatients' or g == 'noofpatients'):
                plt.scatter(data[f],data[g])
            
            elif(f == 'HRG' or g == 'HRG'):
                plt.scatter(data["HRG"], data["nooftreatment"])
                plt.title("Scatter plot of HRG VS Number of Treatments")
                plt.ylabel("Number of Treatments")
                plt.xlabel("HRG")
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                plt.scatter(data[f],data[g])
        
        elif(f == 'noofpatients' or g =='noofpatients'):
            
            if(f == 'HRG' or g == 'HRG'):
                plt.scatter(data["HRG"], data["noofpatients"])
                plt.title("Scatter plot of HRG VS Number of Patients")
                plt.ylabel("Number of Patients")
                plt.xlabel("HRG")
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                plt.scatter(data[f],data[g])
                
        elif(f == 'HRG' or g == 'HRG'): 
            if(f == 'DayofWeek' or g == 'DayofWeek'):
                plt.scatter(data["HRG"], data["DayofWeek"])
                plt.title("Scatter plot of HRG VS Day of the Week")
                plt.ylabel("Day of the Week")
                plt.xlabel("HRG")
        
    elif(h==2): #BAR PLOTS
        
        if(f == 'Age' or g =='Age'):
            if(f == 'LoS' or g == 'LoS'):
                plt.bar(data["Age"], dfsample["LoS"])
                plt.title("LoS by Age")
                plt.ylabel("Length of Stay")
                plt.xlabel("Age")
                
            elif(f == 'noofinvestigation' or g == 'noofinvestigation'):
                a = list(range(0,7))
                b = (df.groupby('noofinvestigation')['Age'].mean())
                plt.bar(a,b, color = "#FF51FF", edgecolor= "#7FFF00")
                plt.title("Age VS No. of Investigations")
                plt.ylabel('Mean Age')
                plt.xlabel('No. of Investigations')
            
            elif(f == 'nooftreatment' or g == 'nooftreatment'):
                plt.bar(data[f],data[g])
            
            elif(f == 'noofpatients' or g == 'noofpatients'):
                plt.bar(data[f],data[g])
            
            elif(f == 'HRG' or g == 'HRG'):
                a = ("VB02Z", "VB03Z",'VB04Z', 'VB05Z', 'VB06Z', 'VB07Z', 'VB08Z', "VB09Z", "VB11Z")
                b = (data.groupby('HRG')['Age'].mean())
                plt.bar(a,b)
                plt.title("Age against HRG")
                plt.ylabel('Age')
                plt.xlabel('HRG code')
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                plt.bar(data[f],data[g])
                
                
        
        elif(f == 'LoS' or g =='LoS'):
            if(f == 'noofinvestigation' or g == 'noofinvestigation'):
                a = list(range(0,7))
                b = (data.groupby('noofinvestigation')['LoS'].mean())
                plt.bar(a,b, color = "#FF51FF", edgecolor = "#FFEFDB")
                plt.title("Mean Length of stay VS No. of Investigations")
                plt.ylabel('Mean Length of stay')
                plt.xlabel('No. of Investigations')
                
            elif(f == 'nooftreatment' or g == 'nooftreatment'):
                a = list(range(0,4))
                b = (data.groupby('nooftreatment')['LoS'].mean())
                plt.bar(a, b, width=0.25)
                plt.title("Mean LoS by nooftreatments")
                plt.ylabel("Length of Stay")
                plt.xlabel("nooftreatments")
            
            elif(f == 'noofpatients' or g == 'noofpatients'):
                a = list(range(0,69))
                plt.bar(data["noofpatients"], dfsample["LoS"])
                plt.title("LoS by noofpatients")
                plt.ylabel("Length of Stay")
                plt.xlabel("noofpatients")
            
            elif(f == 'HRG' or g == 'HRG'):
                a = ("VB02Z", "VB03Z",'VB04Z', 'VB05Z', 'VB06Z', 'VB07Z', 'VB08Z', "VB09Z", "VB11Z")
                b = (data.groupby('HRG')['LoS'].mean())
                plt.bar(a,b)
                plt.title("Length of stay against HRG")
                plt.ylabel('Length of stay')
                plt.xlabel('HRG code')
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                a = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
                b = (data.groupby('DayofWeek')['LoS'].mean())
                plt.bar(a, b, color = ("m"))
                plt.title("Mean LoS by Day of Week")
                plt.ylabel("Length of Stay")
                plt.xlabel("DayofWeek")
                
        
        elif(f == 'noofinvestigation' or g =='noofinvestigation'):
            if(f == 'nooftreatment' or g == 'nooftreatment'):
                a = list(range(0,7))
                b = (data.groupby('noofinvestigation')['nooftreatment'].mean())
                plt.bar(a,b, color = "#FF51FF", edgecolor = "#8A2BE2")
                plt.title("No of treatments VS No. of Investigations")
                plt.ylabel('Average No of Treatments')
                plt.xlabel('No of Investigations')
            
            elif(f == 'noofpatients' or g == 'noofpatients'):
                a = list(range(0,7))
                b = (data.groupby('noofinvestigation')['noofpatients'].mean())
                plt.bar(a,b, color = "#FF51FF", edgecolor = "#00FFFF")
                plt.title("No of Patients VS No. of Investigations")
                plt.ylabel('Average No of Patients')
                plt.xlabel('No of Investigations')
            
            elif(f == 'HRG' or g == 'HRG'):
                a = ("VB02Z", "VB03Z",'VB04Z', 'VB05Z', 'VB06Z', 'VB07Z', 'VB08Z', "VB09Z", "VB11Z")
                b = (data.groupby('HRG')['noofinvestigation'].mean())
                plt.bar(a,b, color = "#FF7F50", edgecolor = "#FF51FF")
                plt.title("No of investigations against HRG")
                plt.ylabel('No of Investigations')
                plt.xlabel('HRG code')
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                a = ("Monday", "tuesdsay",'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
                b = (data.groupby('DayofWeek')['noofinvestigation'].mean())
                plt.bar(a,b, color = "#FF51FF", edgecolor = "#1E90FF")
                plt.title("No of investigations against Day of the Week")
                plt.ylabel('No of Investigations')
                plt.xlabel('Day of the Week')
                
        elif(f == 'nooftreatment' or g =='nooftreatment'):
            
            if(f == 'noofpatients' or g == 'noofpatients'):
                plt.bar(data[f],data[g])# PLOT
            
            elif(f == 'HRG' or g == 'HRG'):
                a = ("VB02Z", "VB03Z",'VB04Z', 'VB05Z', 'VB06Z', 'VB07Z', 'VB08Z', "VB09Z", "VB11Z")
                b = (data.groupby('HRG')['nooftreatment'].mean())
                plt.bar(a,b)
                plt.title("No of treatments against HRG")
                plt.ylabel('No of Treatments')
                plt.xlabel('HRG code')
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                plt.bar(data[f],data[g])# PLOT
        
        elif(f == 'noofpatients' or g =='noofpatients'):
            
            if(f == 'HRG' or g == 'HRG'):
                a = ("VB02Z", "VB03Z",'VB04Z', 'VB05Z', 'VB06Z', 'VB07Z', 'VB08Z', "VB09Z", "VB11Z")
                b = (data.groupby('HRG')['noofpatients'].mean())
                plt.bar(a,b)
                plt.title("No of Patients against HRG")
                plt.ylabel('No of Patients')
                plt.xlabel('HRG code')
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                plt.bar(data[f],data[g])# PLOT
                
        elif(f == 'HRG' or g == 'HRG'): 
            if(f == 'DayofWeek' or g == 'DayofWeek'):
                plt.bar(data[f],data[g])# PLOT
        
        
    elif(h==3):
        return data[f].hist()


# In[ ]:





# In[ ]:


#sample.min


# In[ ]:


#patient_info()


# In[35]:


def main(data = sample):
    m = int(input("choose \n 1. correlate 2. plot 3.range 4. patient info  5. Categorical \n"))
    if(m == 1):
        zzz = int(input("Do you want to correlate 1. Full Data or  2.Two Variables"))
        if(zzz == 1):
            full_correlation(data)
        if(zzz == 2):
            inp1 , inp2 = input("Which two varibles do you want to compare? \n 1: Age   2:LoS   3: noofinvestigation   4: nooftreatments   5:noofpatients 7. Day of the Week  \n").split()
            a, b = ifs(int(inp1), int(inp2))
            print(correlation(a,b,data))
   
    elif(m == 2):
        
        zzz = int(input("Do you want inputs for 1 or 2 variables"))
        if(zzz == 1):
            inp9 = input("Which variable do you want to plot? \n 1: Age 2: LoS 3: noofinvestigation 4: nooftreatment 5:noofpatients 6: HRG\n")
            x = single_ifs(int(inp9))
            inp10 = int(input("Choose one of the following plots:\n 1 : Histogram \n 2 : BoxPlot \n 3 : Pie Chart \n 4 : Dist Plot"))
            plots(x,inp10)
            
        elif(zzz == 2): 
            inp6 = input("Which variable do you want to plot? \n  1: Age 2: LoS   3: noofinvestigation  4: nooftreatment  5:noofpatients  6: HRG 7. Day of the Week\n")
            a = single_ifs(int(inp6))
            v1 = trying(a)
            inp8 = int(input("Choose one of the following plots:\n 1 : Scatter Plot \n 2 : Bar Plot \n 3 : Histogram \n"))
            br = single_ifs(v1)
            plots_simple(a,br,inp8,data)
    
    elif(m == 3):
        inp3 = int(input("Which variable do you want to analyse? \n 1: Age    2: LoS   3: noofinvestigation  4: nooftreatments  6: HRG   5: noofpatients 7. Day of the Week \n"))
        v2 = single_ifs(inp3)
        data_range = variable_range(v2)
        print(data_range)
        
        yn = input("Do you want to anylyse data for the range selected?   (Y/N) ===> ")
        if(yn == 'Y' or yn == 'y'):
            main(data_range)
    
    elif(m == 4):
        Patient_Id = input(" Please provide the id of patient ? ")
        Patient_info = data.loc[sample['ID'] == Patient_Id]
        if Patient_info.empty == True:
            print("Oops! That was no valid number. Try again...")
        else:
            print(Patient_info)
    
    elif(m == 5):
        inp10 = int(input("Which variable do you want to analyse? \n 1: Age   2:LoS   3: noofinvestigation   4: nooftreatments   5:noofpatients  6. HRG   7. Day of the Week \n"))
        v3 = single_ifs(inp10)
        data_range = categorical(v3)
        print(data_range)
        
        yn = input("Do you want to anylyse data for the range selected?   (Y/N) ===> ")
        if(yn == 'Y' or yn == 'y'):
            main(data_range)
        

main()


# In[ ]:


print(len(keys))


# In[ ]:


class Graph:
    
    def __init__(self,xaxis = "Not available", yaxis = "Not available"):
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.type = type
        self.color ='OliveDrab'
        self.alpha = 0.65
        self.rwidth = 0.85
        
    def getBinsList(inputvariable, inputValue):
        if(inputvariable=="Age")
            return np.arange(0, 102, inputValue).tolist()
         
    def PlotHistogram(self):
            n, bins, patches = plt.hist(x=data[xaxis], bins=getBinsList(xaxis, inputValue), color=self.color,
                            alpha=self.alpha, rwidth=self.rwidth)

                plt.grid(axis='y', alpha=0.75)
                plt.xlabel(xaxis)
                plt.ylabel(yaxis)
                plt.title(type+" "+xaxis)
                maxfreq = n.max()
            # Set a clean upper y-axis limit.
                plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
                


# In[ ]:




