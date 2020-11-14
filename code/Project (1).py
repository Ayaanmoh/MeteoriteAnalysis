import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

pop = pd.read_csv('C:/Users/Admin/Downloads/ALDA522/Project/results.csv')#,dtype={'lon': np.float64, 'lat': np.float64, 'size': np.float64, 'population': np.float64,'area': np.float64})
pop=pop.dropna()#=pop[pop.isnull() == False]
pop['fallCode'] = pd.Categorical(pd.factorize(pop['fall'])[0])
data = pop[['recclass','mass','reclat','reclong','pop_lat','pop_lon','population']]
labels = pop[['fallCode']]

train,test,trainLabel,testlabel= train_test_split(data,labels,stratify=labels)
# print(train)
model = Sequential()
model.add(Dense(15, input_dim=4, activation='relu'))
# model.add(Dense(60, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(train[['mass','population','reclat','reclong']], trainLabel, epochs=5, batch_size=10,verbose=1)
_, accuracy = model.evaluate(test[['mass','population','reclat','reclong']], testlabel)
print(accuracy)
# # pop=pop[pop['lat']<=-56]
# # print(pop)
# pop_lat = []
# pop_lon = []
# size = []
# population = []
# area = []
# for idx,sighting in met.iterrows():
#     if idx%1000 == 0:
#         print(idx, 31686, time.time())
#     # print(sighting[['reclat', 'reclong']])
#     tmp = pop.loc[pop['lat']<=sighting['reclat']].loc[pop['lon']<=sighting['reclong']]
#     # print(tmp.size)
#     # tmp = tmp[tmp['lon']<=sighting['reclong']]
#     # print(tmp.size)
#     a=(tmp[['lat', 'lon']].sub(np.array(sighting[['reclat', 'reclong']])).pow(2).sum(1).pow(0.5)).argsort()
#     # print(a)
#     # print(a[0])
#     tmp = tmp.iloc[a[:1]]
#     if tmp.size == 0:
#         pop_lat.append(0)
#         pop_lon.append(0)
#         size.append(0)
#         population.append(0)
#         area.append(0)
#         continue

#     # print(tmp.size)
#     # print(tmp[['lat', 'lon']],'\n')
#     # print(tmp['lat'])
#     pop_lat.append(tmp['lat'].values[0])
#     pop_lon.append(tmp['lon'].values[0])
#     size.append(tmp['size'].values[0])
#     population.append(tmp['population'].values[0])
#     area.append(tmp['area'].values[0])
#     # sighting['pop_lat'] = tmp['lat']
#     # sighting['pop_lat'],sighting['pop_lon'],sighting['size'],sighting['population'],sighting['area'] = tmp['lat'],tmp['lon'],tmp['size'],tmp['population'],tmp['area']
#     # print(sighting)
#     # if idx > 10:
#     #     break

# met['pop_lat'],met['pop_lon'],met['size'],met['population'],met['area']=pop_lat,pop_lon,size,population,area
# print(met.iloc[:10])
# met.to_csv('C:/Users/Admin/Downloads/ALDA522/Project/results.csv')

# # def getValue(entry):
# #     print(entry)
# #     a=entry.split(' ')
# #     int(a[5])

# # res = pd.read_csv('C:/Users/Admin/Downloads/ALDA522/Project/results.csv')
# # print(res['pop_lat'].str.split(" ")[5])
# # res = res['pop_lat'].apply(getValue)
# # res = res['pop_lon'].apply(getValue)
# # res = res['size'].apply(getValue)
# # res = res['population'].apply(getValue)
# # res = res['area'].apply(getValue)

# # print(res.iloc[:10])
