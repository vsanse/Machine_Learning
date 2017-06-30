import numpy as np 
import pandas as pd
import random
from collections import Counter as co
def k_nearest_neighbors(data, predict, k = 3):
    if len(data) > k:
        warnings.warn('K is set to a value less than total voting groups. Idiot!')
    distance = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distance.append([euclidean_distance,group]) 
    votes = [i[1] for i in sorted(distance)[:k]]
    vote_result = co(votes).most_common(1)[0][0] 
    confidence = co(votes).most_common(1)[0][1] /k  
    # print(co(votes))
    print(confidence)
    return vote_result, confidence
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace = True)
#removing useless columns like id
df.drop(['id'], 1, inplace = True) 
#print (df)
#convertimg to float becauz for some reason data is converted to quotes thus treated as string

full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2

train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])    

# print(train_set)
# print(20*'#')

# print (test_set )

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote,confidence = k_nearest_neighbors(train_set,data, k = 5)
        if group == vote:
            correct += 1
        total += 1    
print('Accuracy: ',correct/total)        

