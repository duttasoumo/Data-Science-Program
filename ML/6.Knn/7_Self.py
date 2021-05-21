import csv, random, math

def euc(obj1, obj2, size):
    dist= 0
    for x in range(size):
        dist= dist+pow((obj1[x]-obj2[x]), 2)
    return math.sqrt(dist)

def gen_nbors(tr_data, te_data, k):
    dist= []; nbors= []; c= (len(te_data))-1
    for x in range(len(tr_data)):
        d= euc(te_data, tr_data[x], c)
        dist.append((tr_data[x], d))
    dist= sorted(dist, key= lambda x: x[1])
    for x in range(k):
        nbors.append(dist[x][0])
    return nbors
    
def gen_response(nbors):
    temp= {}
    for x in range(k):
        pred= nbors[x][-1]
        if pred in temp:
            temp[pred]= temp[pred]+1
        else:
            temp[pred]= 1
    sorted_pred= list(temp.items()); #print(temp, list(temp.items()))
    return sorted_pred[0][0]
    
file= open('iris.data', 'r'); dataset= list(csv.reader(file))
tr_data= []; te_data= []
te_size= input("Enter the training data size: "); te_size= float(te_size)
k= input("Enter the value of k: "); k= int(k)
#k= math.sqrt(int(len(tr_data))); print(k); k= int(k); print(k)

for x in range(len(dataset)-1):
        for y in range(len(dataset[0])-1):
            dataset[x][y]= float(dataset[x][y])
        if random.random() < float(te_size):
            tr_data.append(dataset[x])
        else:
            te_data.append(dataset[x])

rt= 0; out= []
print('\nNumber of training data samples: '+str(len(tr_data)))
#print('\nACTUAL CLASS\t\t\tPREDICTED CLASS')

for x in range(len(te_data)):
    nbors= gen_nbors(tr_data, te_data[x], k); predd= gen_response(nbors)
    out.append(predd)
    #print(str(te_data[x][-1])+'\t\t\t\t\t\t'+str(predd)) if(predd==te_data[x][-1]) else print(str(te_data[x][-1])+'\t\t\t\t'+str(predd)+' <== Prediction inaccurate.') 

for x in range(len(te_data)):
    if te_data[x][-1]== out[x]:
        rt= rt+1
acc= (rt/float(len(te_data)))
print('\nPrediction accuracy: '+str(acc))
