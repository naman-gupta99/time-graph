from collections import defaultdict
import csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set_style('darkgrid')

class Entry:
    def __init__(self, startDateTime=None):
        self.startDateTime = startDateTime
        self.durations = []

def getDate(s):
    return datetime.strptime(s, '%Y-%m-%d')

def getTime(s):
    return datetime.strptime(s, '%H:%M:%S')

def getWorkableData(rows):
    res = defaultdict(Entry)
    for row in rows:
        startDateTime = getDate(row[7])
        duration = getTime(row[11])
        res[startDateTime].startDateTime = startDateTime
        res[startDateTime].durations.append(duration)
    return list(res.values())

def getPredictedTime(data):
    x = []
    y = []
    last = 0
    for entry in data:
        y.append(entry.startDateTime.toordinal() - data[0].startDateTime.toordinal())
        duration = sum(timedelta(hours=i.hour, minutes=i.minute, seconds=i.second).total_seconds()/3600 for i in entry.durations)
        x.append(last + duration)
        last += duration
    # print(x, y)
    x = np.array(x)
    y = np.array(y)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    res_feature = [60]
    res_feature = np.array(res_feature)
    res_feature = res_feature.reshape(-1, 1)
    return model.predict(res_feature)

def makeGraph(data, final_date):
    x = []
    y = []
    last = 0
    for entry in data:
        x.append(entry.startDateTime)
        duration = sum(timedelta(hours=i.hour, minutes=i.minute, seconds=i.second).total_seconds()/3600 for i in entry.durations)
        y.append(last + duration)
        last += duration
    plt.figure(figsize=(10,2), tight_layout=True)
    sns.lineplot(x=x, y=y, linestyle='-')
    x1 = [getDate('2022-06-13'), final_date]
    y1 = [60, 60]
    sns.lineplot(x=x1, y=y1, linestyle='dashed')
    x2 = [getDate('2022-06-13'), final_date]
    y2 = [0, 60]
    sns.lineplot(x=x2, y=y2, linestyle='dashed')
    plt.show()

file = open('data.csv', 'r')
reader = csv.reader(file)
header = next(reader)
rows = []
for row in reader:
    rows.append(row)
file.close()

workableData = getWorkableData(rows)

prediction = getPredictedTime(workableData)
prediction = prediction[0][0]
prediction = int(prediction) if prediction % 1 < 0.5 else int(prediction) + 1

print(datetime.fromordinal(prediction + workableData[0].startDateTime.toordinal()))

makeGraph(workableData, datetime.fromordinal(prediction + workableData[0].startDateTime.toordinal()))