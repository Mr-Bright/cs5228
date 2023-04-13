# author by Li Xingchen
# Date 2023/3/20 22:30

import pandas as pd
from geopy.distance import distance

def getProcessedDataset(dataset1,datasetName2) :

  dataset2 = pd.read_csv('auxiliary-data/'+datasetName2+'.csv')

  # Create new Series
  nearestDistanceColName = datasetName2+'_nearestDistance/KM'
  nearestDistanceCol = pd.Series(name=nearestDistanceColName)
  lessHalfKMNumColName  = datasetName2+'_lessHalfKMNum'
  lessHalfKMNumCol = pd.Series(name=lessHalfKMNumColName)
  half2OneKMNumColName  = datasetName2+'_half2OneKMNum'
  half2OneKMNumCol = pd.Series(name=half2OneKMNumColName)
  one2ThreeKMNumColName  = datasetName2+'_one2ThreeKMNum'
  one2ThreeKMNumCol = pd.Series(name=one2ThreeKMNumColName)

  for index1,row in dataset1.iterrows() :
    #print('-------------'+'train1'+'_'+str(index1)+'_begin'+'-------------')

    dataset1_lat = row['latitude']
    dataset1_lng = row['longitude']

    dataset1_location = (dataset1_lat,dataset1_lng)

    nearestDistance = 99999999999999999999.99
    lessHalfKMNum = 0
    half2OneKMNum = 0
    one2ThreeKMNum = 0

    for index,row in dataset2.iterrows() :
      #print('========'+datasetName2+'_'+str(index)+'_begin'+'========')

      dataset2_lat = row['lat']
      dataset2_lng = row['lng']

      dataset2_location = (dataset2_lat,dataset2_lng)

      distance_between = distance(dataset1_location, dataset2_location).km

      if distance_between < nearestDistance :
        nearestDistance = distance_between

      if distance_between < 0.5 :
        lessHalfKMNum += 1
      elif distance_between < 1 :
        half2OneKMNum += 1
      elif distance_between < 3 :
        one2ThreeKMNum += 1

      #print('nearestDistance',nearestDistance)
      #print('lessHalfKMNum',lessHalfKMNum)
      #print('half2OneKMNum',half2OneKMNum)
      #print('one2ThreeKMNum',half2OneKMNum)
      #print('========'+datasetName2+'_'+str(index)+'_end'+'========')

    #print('nearestDistance',nearestDistance)
    #print('lessHalfKMNum',lessHalfKMNum)
    #print('half2OneKMNum',half2OneKMNum)
    #print('one2ThreeKMNum',half2OneKMNum)

    nearestDistanceCol.loc[index1] = nearestDistance
    lessHalfKMNumCol.loc[index1] = lessHalfKMNum
    half2OneKMNumCol.loc[index1] = half2OneKMNum
    one2ThreeKMNumCol.loc[index1] = one2ThreeKMNum

    print('-------------'+'train1'+'_'+str(index1)+'_end'+'-------------')

  dataset1 = pd.concat([dataset1, nearestDistanceCol], axis=1)
  dataset1 = pd.concat([dataset1, lessHalfKMNumCol], axis=1)
  dataset1 = pd.concat([dataset1, half2OneKMNumCol], axis=1)
  dataset1 = pd.concat([dataset1, one2ThreeKMNumCol], axis=1)

  return dataset1

def concatAdditionalInfo(dataset) :

  result = getProcessedDataset(dataset,'sg-primary-schools')
  #result = getProcessedDataset(result,'sg-commerical-centres')
  #result = getProcessedDataset(result,'sg-secondary-schools')
  #result = getProcessedDataset(result,'sg-shopping-malls')
  #result = getProcessedDataset(result,'sg-train-stations')
  #result = getProcessedDataset(result,'sg-gov-markets-hawker-centres')

  result.to_csv('result.csv', index=False)
  #return result

if __name__=='__main__':
    #dataset = pd.read_csv('train.csv')
    #concatAdditionalInfo(dataset)
    result = pd.read_csv('result.csv')
    concatAdditionalInfo(result)

##22:32
