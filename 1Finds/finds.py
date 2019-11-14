
import random
import csv
attributes = [['Sunny','Rainy'],
              ['Warm','Cold'],
              ['Normal','High'],
              ['Strong','Weak'],
              ['Warm','Cool'],
              ['Same','Change']]

num_attributes = len(attributes)

print (" \n The most general hypothesis : ['?','?','?','?','?','?']\n")
print ("\n The most specific hypothesis : ['0','0','0','0','0','0']\n")

a = []
print("\n The Given Training Data Set \n")
with open('C:\\Users\\Gunjan\\Desktop\\ML_Dataset\\FindS\\enjoysport.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        a.append (row)
        print(row)

print("\n The initial value of hypothesis: ")
hypothesis = ['0'] * num_attributes
print(hypothesis)

# Comparing with First Training Example 
for j in range(0,num_attributes):
        hypothesis[j] = a[0][j];

# Comparing with Remaining Training Examples of Given Data Set
print("\n Find S: Finding a Maximally Specific Hypothesis\n")

for i in range(0,len(a)):
    if a[i][num_attributes]=='Yes':
            for j in range(0,num_attributes):
                if a[i][j]!=hypothesis[j]:
                    hypothesis[j]='?'
                else :
                    hypothesis[j]= a[i][j] 
    print(" For Training Example No :{0} the hypothesis is ".format(i),hypothesis)
                
print("\n The Maximally Specific Hypothesis for a given Training Examples :\n")
print(hypothesis)


