import pandas as pd
import numpy as np
import time

import os
dir='C:/Users/Admin/Downloads/ALDA522/Project/exp/sedac-gpw-parser/sedac_gpw_parser'
fns = os.listdir(dir+'/popDataSorted')

# pop = pd.read_csv('C:/Users/Admin/Downloads/ALDA522/Project/gpwv4-2015.csv')#,dtype={'lon': np.float64, 'lat': np.float64, 'size': np.float64, 'population': np.float64,'area': np.float64})
met = pd.read_csv('C:/Users/Admin/Downloads/ALDA522/Project/mdata_primary.csv')#,dtype={'reclon': np.float64, 'reclat': np.float64})
# pop=pop[pop['lat']<=-56]
# print(pop)
pop_lat = []
pop_lon = []
size = []
population = []
area = []
for idx,sighting in met.iterrows():
    if idx%1000 == 0 or True:
        print(idx, 31686, time.time())
    # print(sighting[['reclat', 'reclong']])
    df=pd.DataFrame()
    for fn in fns:
        pop = pd.read_csv(dir+'/popDataSorted/'+fn)
        tmp = pop.loc[pop['lat']<=sighting['reclat']].loc[pop['lon']<=sighting['reclong']]
        if not tmp[-1].isnull():
            df.append(tmp[-1])

    # print(tmp.size)
    # tmp = tmp[tmp['lon']<=sighting['reclong']]
    # print(tmp.size)
    a=(df[['lat', 'lon']].sub(np.array(sighting[['reclat', 'reclong']])).pow(2).sum(1).pow(0.5)).argsort()
    # print(a)
    # print(a[0])
    df = df.iloc[a[:1]]
    # if df.size == 0:
    #     pop_lat.append(0)
    #     pop_lon.append(0)
    #     size.append(0)
    #     population.append(0)
    #     area.append(0)
    #     continue

    # print(df.size)
    # print(df[['lat', 'lon']],'\n')
    # print(df['lat'])
    pop_lat.append(df['lat'].values[0])
    pop_lon.append(df['lon'].values[0])
    # size.append(df['size'].values[0])
    population.append(df['population'].values[0])
    # area.append(df['area'].values[0])
    # sighting['pop_lat'] = df['lat']
    # sighting['pop_lat'],sighting['pop_lon'],sighting['size'],sighting['population'],sighting['area'] = df['lat'],df['lon'],df['size'],df['population'],df['area']
    # print(sighting)
    if idx > 10:
        break

met['pop_lat'],met['pop_lon'],met['population']=pop_lat,pop_lon,population
print(met.iloc[:10])
# met.to_csv('C:/Users/Admin/Downloads/ALDA522/Project/results.csv')

# def getValue(entry):
#     print(entry)
#     a=entry.split(' ')
#     int(a[5])

# res = pd.read_csv('C:/Users/Admin/Downloads/ALDA522/Project/results.csv')
# print(res['pop_lat'].str.split(" ")[5])
# res = res['pop_lat'].apply(getValue)
# res = res['pop_lon'].apply(getValue)
# res = res['size'].apply(getValue)
# res = res['population'].apply(getValue)
# res = res['area'].apply(getValue)

# print(res.iloc[:10])
