
import csv
# with open('copy_csv.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     eps_axis = []
#     acc_axis = []
#     for row in spamreader:
#         for i,obj in  enumerate(row):
#             if obj.__contains__("EPS:"):
#                 term = row[i+1].split(",")
#                 eps_axis.append(float(term[0]))
#             if obj.__contains__("Test:"):
#                 term = row[i+1]
#                 acc_axis.append(float(term))
#         print(row)
#     print(eps_axis)
#     print(acc_axis)

with open('copy_csv.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    eps_axis = []
    acc_axis = []
    for row in spamreader:
        for i,obj in  enumerate(row):
            if obj.__contains__("EPS:"):
                term = row[i+1].split(",")
                eps_axis.append(float(term[0]))
            if obj.__contains__("Acc"):
                term = row[i+1]
                acc_axis.append(float(term))
        print(row)
    print(eps_axis)
    print(acc_axis)


import matplotlib.pyplot as plt
import numpy as np

eps_axis.reverse()
acc_axis.reverse()

eps_axis = eps_axis[:-1]
acc_axis = acc_axis[:-1]

plt.xticks(np.arange(min(eps_axis), max(eps_axis)+1, 200.0))


plt.plot(eps_axis, acc_axis)
plt.title('DP-KE-GCN on FB15K')
plt.xlabel('Epsilon Values')
plt.ylabel('Test Acurracy')
plt.show()