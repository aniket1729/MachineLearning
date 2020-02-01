#!/usr/bin/env python
# coding: utf-8

# In[54]:


# Libraries required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn


# In[55]:


# Symbol Table
print(dir())
import mathlib
print(dir(mathlib))


# In[56]:


mathlib.add(5, 6)


# In[57]:


import sys
print(sys.path)


# In[58]:


import mathlib
ob1 = mathlib.Adder(5,6)
ob1.add()


# In[59]:


# NumPy array operation
import numpy as np
p1 = np.array([60,60,3,1,0,0])
p2 = p1 * 2
print(p1)
print(p2)
print(p2.sum())


# In[60]:


# NumPy ndarray operation
import numpy as np
p1 = np.ndarray([60,60,3,1,0,0])
p2 = p1 * 2
print(p1)
print(p2)
print(p2.sum())


# In[61]:


range(1,11)
range(11)

print("Range op")
print(list(range(1,11)))
print(list(range(11)))

import numpy as np
np.arange(1,10,0.5)
print("Numpy op")
print(np.arange(1,10,0.5))
print(np.arange(1,10,1))
print(np.arange(1,11,1))


# In[62]:


# NumPy array operation faster than Python loop operation over array
import time   # Vector

start = time.time()
a = list(range(1,1000001))
b = list(range(1,1000001))
c1 = []
for i in range(len(a)):
 c1.append(a[i]*b[i])
end = time.time()
print("time taken = ",end-start)

import numpy as np
start = time.time()
a = np.array(np.arange(1,1000001))
b = np.array(np.arange(1,1000001))
c2 = a*b
end = time.time()
print("time taken = ",end-start)

print("First = ",c1[:10])
print("Secnd = ",c2[:10])


# In[63]:


# Type conversion, Data Type, Dimension and Shape of array
import numpy as np

v1 = np.array([50,30])

print(v1)
print(v1.ndim)
print(v1.shape)
print(v1.dtype)
print("")

v2 = v1 * 1.0
print(v2)
print("")

v3 = v1.astype("double")
print(v3)
print(v3.dtype)
v3 = v1.astype(np.float32)
print(v3)
print(v3.dtype)
v3 = v1.astype(np.object)
print(v3)
print(v3.dtype)


# In[64]:


# Changing array shape
print(v1.reshape(1, -1))
print()
print(v1.reshape(-1, 1))


# In[65]:


# File operation
fob = open("file-write-read.txt","w")
fob.write("10\n32\n41\n73\n25\n4")
fob.close()

fob = open("file-write-read.txt", "r")
numlst = fob.readlines()
fob.close()

# python - strarray into an - int array
numlst[:] = [int(e) for e in numlst]  
print(numlst)
print()

# Condition checking on all array elements
import numpy as np
vec1 = np.array(numlst)
print(vec1[ vec1 > 50])
print(vec1[ vec1 <= 50])
print()

print(vec1[ vec1%2 ==0])
print(vec1[ vec1%2 != 0])
print()


# In[66]:


# SciPy library functions
from scipy import stats

print(vec1.mean())
print(np.median(vec1))    #library function
print(stats.mode(vec1))    #library function


# In[67]:


print(vec1**2)
print(vec1.cumsum())
print(vec1+2)
vec1[0]= 20
print(20 in vec1)


# In[68]:


# Pandas DataFrame creation
state = ["kar", "tn", "ts", "ker","ap"]
city  = ["blr", "chn", "hyd", "tvm",np.NaN]

df = pd.DataFrame( {
                    "col1" : state,
                    "col2" : city 
                   } 
                 )
print(df)


# In[69]:


df.columns = ["state", "city"]    # DataFrame column labeling / renaming
print(df)
df


# In[70]:


#Adding new column to DataFrame
df["population"] = [50, 60, 70, 80, 90]
df


# In[71]:


# DataFrame selection of rows
print(df[ df.population >= 60 ])
print()

print(df.population[ df.population >= 60 ])
print()

print(df["population"][ df.population >= 60 ])


# In[72]:


# Plotting DataFrame
df["edu"] = [5,3,2,4,1]
df.plot(x="population", y="edu")
plt.show()


# In[73]:


csv_file = open("data-all.csv","w")
csv_file.write("test1:10:20\ntest2:20:20\ntest3:50:50\ntest4:15:15\ntest5:25:20\ntest6::30\ntest7::\ntest8:5:")
csv_file.close()

# Creating DataFrame from CSV file
csv_df = pd.read_csv("data-all.csv", header=None, delimiter=":")
#csv_df = pd.read_csv(filename="data.csv", header=None, delimiter="," sep=",", encoding="utf-8", name=["name", "exp", "got"])

csv_df.columns = ["name", "exp", "got"]
csv_df.fillna(0, inplace=True)

csv_df["diff"] = csv_df["exp"] - csv_df["got"]

#csv_df["status"] = "Pass" if csv_df["diff"] >= 0 else "Fail"
csv_df["status"] = np.where(csv_df["diff"] >= 0,"Pass", "Fail")
print(csv_df)

plt.bar(csv_df["status"], height=csv_df["exp"])

#csv_df.plot()

#plt.bar(csv_df["name"], csv_df["status"])

#plt.bar(range(1,len(csv_df["status"])+1), csv_df["status"])
plt.show()


# In[74]:


csv_df1 = csv_df[csv_df.status == "Pass"]
csv_df2 = csv_df[csv_df.status != "Pass"]
#csv_df2 = csv_df[~csv_df.status == "Pass"]
#csv_df2 = csv_df[csv_df.status == "Fail"]

# Writing DataFrame to CSV file
csv_df1.to_csv("data-pass.csv", index=False);
csv_df2.to_csv("data-fail.csv", index=False);


# In[75]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv("data-pass.csv")
df2 = pd.read_csv("data-fail.csv")

#print df1
#print df2

#inner join = A intersection B
#outer join = A union B
#left outer join  = A + A intersection B
#right outer join = B + A intersection B

# Merging 2 DataFrames
df3 = pd.merge(df1,df2,how="outer",on=["name","exp","got"])
print(df3)
df3.fillna(value="sales",inplace=True)
print(df3)


# In[76]:


emp1X = open("emp-data1.xlsx","w")
emp1X.write("empid name dept salary\n1234 arun sales 45000\n1235 john finan 56000\n1236 ravi purch 65656\n1237 manu accts 45454\n1238 raja hrd 43434")
emp1X.close()

emp2C = open("emp-data2.csv", "w")
emp2C.write("1234,blr,band5\n1231,chn,band4\n1236,hyd,band1\n1230,tvm,band3\n1232,mum,band2")
#emp2C.write("code,loc,band\n1234,blr,band5\n1231,chn,band4\n1236,hyd,band1\n1230,tvm,band3\n1232,mum,band2")
emp2C.close();

lblT = open("emp-labels.txt", "w")
lblT.write("code,loc,band")
lblT.close();


emp1_df = pd.read_csv("emp-data1.xlsx", header=0, delimiter=" ")
#pd.read_excel("emp1.xlsx", sheet="sheet1")

emp2_df = pd.read_csv("emp-data2.csv", header=None)

lbl_fp = open("emp-labels.txt")
lbl_col = lbl_fp.read().split(",")
lbl_fp.close()

emp2_df.columns = lbl_col

merge_df = pd.merge(emp1_df, emp2_df, how="inner", left_on = "empid", right_on = "code")
#merge_df = pd.merge(emp1_df, emp2_df.set_index("code"), how="inner", left_on = "empid", right_on = "code")
print(merge_df)

#out.csv  - empid,name,loc,salary
merge_df.to_csv("emp-out.csv", columns= ["empid", "name", "loc", "salary"], index=False);
merge_df[["empid", "name", "loc", "salary"]].to_csv("emp-out-another.csv", index=False)


# In[ ]:




