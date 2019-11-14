import numpy as np
import pandas as pd
data=pd.DataFrame(data=pd. read_csv('c1.csv'))
concepts=np.array(data.iloc[:,0:-1])
target=np.array(data.iloc[:,-1])
def  learn(concepts, target):
    specific_h=concepts[0].copy()
    general_h=[["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    for i,h in enumerate(concepts):
        if target[i]=="Y":
            for x in range(len(specific_h)):
                if h[x]!=specific_h[x]:
                    specific_h[x]='?'
                    general_h[x][x]='?'
        if target[i]=="N":
            for x in range(len(specific_h)):
                if h[x]!=specific_h[x]:
                    general_h[x][x]=specific_h[x]
                else:
                    general_h[x][x]='?'
                    
    indices=[i for i,val in enumerate(general_h) if val == ['?','?','?','?','?','?']]
    
    for i in indices:
        general_h.remove(['?','?','?','?','?','?'])
    return specific_h,general_h
s_final,g_final=learn(concepts,target)
print("Final Specific_h:",s_final, sep="\n")
print("Final General_h:",g_final, sep="\n")
