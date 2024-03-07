import csv
num_attributes=6
a=[]
print("\n The Given Training Data Set \n")

with open('enjoysport.csv','r') as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        a.append(row)
        print(row)

print("\n The initial value of hypothesis: ")
hypothesis =['0']*num_attributes
print(hypothesis)

hypothesis=[a[0][j] for j in range(0,num_attributes)]
print("\n Find S: Finding a Maximally Specific Hypothesis\n")

for i in range(0,len(a)):
    if a[i][num_attributes]=='yes':
        hypothesis=['?' if a[i][j]!=hypothesis[j] else hypothesis[j] for j in range(0,num_attributes)]
    print("For Training instance No:{0} the hypothesis is ".format(i),hypothesis)
    
print("\n The Maximally Specific Hypothesis for a given Training Examples:\n")
print(hypothesis)
