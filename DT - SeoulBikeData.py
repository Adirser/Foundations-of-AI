
import pandas as pd
import numpy as np
import datetime as dtt
import pptree

class DecisionTree():
    def __init__(self, depth=0, max_depth=10, features=None,father_values = None,leaf_node = False):
        self.left = None
        self.right = None
        self.fkey = None
        self.fval = None
        self.depth = depth
        self.max_depth = max_depth
        self.target = None
        self.side = None
        self.features = features
        self.children = []
        self.leaf_node = leaf_node
        self.father_values = father_values

    def __str__(self):
        print = ''
        if self.father_values is not None:
            print = self.father_values
        if not self.leaf_node:
            return print + '--' + self.fkey
        elif self.target == 1:
            return print + '---> Busy'
        else:
            return print + '---> Not Busy'


    def train(self, x_train,father_v = None):
        features = list(self.features)
        info_gains = []
        for ix in features:
            i_gain = calc_information_gain(x_train, ix, 1)
            info_gains.append(i_gain)
        self.fkey = features[np.argmax(info_gains)]  # split features based on best information gain
        self.fval = 1
        real_features = features
        real_features.remove(self.fkey)
        data_right, data_left = divide_data(x_train, self.fkey, self.fval)
        data_right = data_right.reset_index(drop=True)  # Reseeting Indexes for recursion
        data_left = data_left.reset_index(drop=True)  # Reseeting Indexes for recursion
        if data_left.shape[0] == 0 or data_right.shape[0] == 0:  # Target answer is what appears the most
            self.target = np.bincount(x_train["Rented Bike Count"]).argmax()
            self.leaf_node = True
            return
        elif self.depth >= self.max_depth:  # if too deep
            self.target = np.bincount(x_train["Rented Bike Count"]).argmax()
            self.leaf_node = True
            return
        elif len(x_train['Rented Bike Count'].unique()) == 1:  # Theres one answer only.
            self.target = x_train["Rented Bike Count"].unique()[0]
            self.leaf_node = True
            return
        elif len(features) == 0:  # if there are no more attributes - return the majority_value
            self.target = np.bincount(x_train["Rented Bike Count"]).argmax()
            self.leaf_node = True
            return
        else:
            self.left = DecisionTree(self.depth + 1, self.max_depth, real_features,father_values=father_v)
            self.left.train(data_left,father_v=str(0))
            self.children.append(self.left)
            self.right = DecisionTree(self.depth + 1, self.max_depth, real_features,father_values=father_v)
            self.right.train(data_right,father_v =str(1))
            self.children.append(self.right)
            return

    def predict(self, test):
        if test[self.fkey] >= self.fval:
            if self.right is None:
                return self.target
            else:
                return self.right.predict(test)
        if test[self.fkey] < self.fval:
            if self.left is None:
                return self.target
            else:
                return self.left.predict(test)
    def predict_row(self, row,counter = 0):
        if row[counter] >= self.fval:
            if self.right is None:
                return self.target
            else:
                return self.right.predict_row(row,counter+1)
        if row[counter] < self.fval:
            if self.left is None:
                return self.target
            else:
                return self.left.predict_row(row,counter+1)

def bucketing(sbdf):
    for i in range(len(sbdf)):
        date_object = dtt.datetime.strptime(sbdf.loc[i, "Date"], "%d/%m/%Y")
        date_object = date_object.weekday()
        # Weekend of MidWeek ?
        if date_object >= 1 and date_object <= 5:
            sbdf.iloc[i, 0] = 0
        else:
            sbdf.iloc[i, 0] = 1
    sbdf['Hour'] = sbdf['Hour'].apply(
        lambda x: 0 if x >= 6 and x <= 18 else
        1)
    sbdf['Rented Bike Count'] = sbdf['Rented Bike Count'].apply(
        lambda x: 0 if x < 650 else 1)
    sbdf['Temperature(°C)'] = sbdf['Temperature(°C)'].apply(
        lambda x: 0 if x <= 12 else 1)
    sbdf['Humidity(%)'] = sbdf['Humidity(%)'].apply(
        lambda x: 0 if x <= 58 else 1)
    sbdf['Wind speed (m/s)'] = sbdf['Wind speed (m/s)'].apply(
        lambda x: 0 if x <= 2.8 else 1)
    sbdf['Visibility (10m)'] = sbdf['Visibility (10m)'].apply(
        lambda x: 0 if x <= 1436 else 1)
    sbdf['Dew point temperature(°C)'] = sbdf['Dew point temperature(°C)'].apply(
        lambda x: 0 if x < 0 else 1)
    sbdf['Solar Radiation (MJ/m2)'] = sbdf['Solar Radiation (MJ/m2)'].apply(
        lambda x: 0 if x <= 1.4 else 1)
    sbdf['Rainfall(mm)'] = sbdf['Rainfall(mm)'].apply(
        lambda x: 0 if x <= 1.2 else 1)
    sbdf['Snowfall (cm)'] = sbdf['Snowfall (cm)'].apply(
        lambda x: 0 if x <= 0.5 else 1)
    sbdf['Seasons'] = sbdf['Seasons'].apply(
        lambda x: 1 if x == "Summer" or x == "Spring" else 0)
    sbdf['Holiday'] = sbdf['Holiday'].apply(
        lambda x: 0 if x == "No Holiday" else 1)
    sbdf['Functioning Day'] = sbdf['Functioning Day'].apply(
        lambda x: 1 if x == "Yes" else 0)
def calc_entropy(column):
    probablities = np.bincount(column) / len(column)
    entropy = 0
    for prob in probablities:
        if prob > 0.0001:
            entropy += (-1.0 * prob * np.log2(prob))
    return entropy
def divide_data(x_data, fkey, fval):
    # fkey is column name
    # fval is the labels
    x_right = x_data[x_data[fkey] == fval]
    x_left = x_data[x_data[fkey] == 0]
    return x_right,x_left
def calc_information_gain(x_data, fkey, fval):
    right, left = divide_data(x_data, fkey, fval)
    l = float(left.shape[0] / x_data.shape[0])
    r = float(right.shape[0] / x_data.shape[0])
    if left.shape[0] == 0 or right.shape[0] == 0:
        return -99999
    info_gain = calc_entropy(x_data["Rented Bike Count"]) - ((l * calc_entropy(left["Rented Bike Count"])) + (r * calc_entropy(right["Rented Bike Count"])))
    return info_gain
def build_tree(k):
    x_train = df.sample(frac=k)  # random state is a seed value
    x_test = df.drop(x_train.index)
    x_test = x_test.reset_index(drop=True)
    x_train = x_test.reset_index(drop=True)
    features = list(x_train.columns)
    features.remove("Rented Bike Count")
    dt = DecisionTree(features=features)
    dt.train(x_train)
    y_pred = []
    for ix in range(x_test.shape[0]):
        x = dt.predict(x_test.loc[ix])
        y_pred.append(x)
    print(1-np.mean(y_pred == x_test["Rented Bike Count"]), "<<<Error Rate<<<")
    pptree.print_tree(dt,childattr='children')
def tree_error(k):
    accuracy = 0
    for i in range(0, k):
        x_train = df.sample(frac=(k-1)/k)  # random state is a seed value
        x_test = df.drop(x_train.index)
        x_test = x_test.reset_index(drop=True)
        x_train = x_train.reset_index(drop=True)
        features = list(x_train.columns)
        features.remove("Rented Bike Count")
        dt = DecisionTree(features=features)
        dt.train(x_train)
        y_pred = []
        for ix in range(x_test.shape[0]):
            x = dt.predict(x_test.loc[ix])
            y_pred.append(x)
        accuracy += np.mean(y_pred == x_test["Rented Bike Count"])
    avg_accuracy = accuracy/k
    print(avg_accuracy , " is the AVG Accuracy Score")
    print(1-avg_accuracy , "This is the AVG Error Rate")
def bucket_single_row(row_input):
    bucketed = []
    date_string = row_input[0]
    date_obj = dtt.datetime.strptime(date_string,'%d/%m/%Y')
    day = date_obj.weekday()
    if (day >= 1 and day <= 5):
        day = 0
    else:
        day=1
    bucketed.append(day)
    hour = row_input[1]
    if hour >= 6 and hour <= 18:
        hour = 0
    else:
        hour = 1
    bucketed.append(hour)
    temperature = row_input[2]
    if temperature <= 12:
        temperature = 0
    else:
        temperature = 1
    bucketed.append(temperature)
    humidity = row_input[3]
    if humidity <= 58:
        humidity = 0
    else:
        humidity=1
    bucketed.append(humidity)
    wind = row_input[4]
    if wind <= 2.8:
        wind = 0
    else:
        wind = 1
    bucketed.append(wind)
    visibility = row_input[5]
    if visibility <= 1436:
        visibility =0
    else:
        visibility =1
    bucketed.append(visibility)
    dpt =row_input[6]
    if dpt <=0:
        dpt = 0
    else:
        dpt =1
    bucketed.append(dpt)
    sr = row_input[7]
    if sr <= 1.4:
        sr = 0
    else:
        sr = 1
    bucketed.append(sr)
    rain = row_input[8]
    if rain <= 1.2:
        rain = 0
    else:
        rain = 1
    bucketed.append(rain)
    snow = row_input[9]
    if snow <= 0.5:
        snow = 0
    else:
        snow = 1
    bucketed.append(snow)
    season = row_input[10]
    if season =="Summer" or season == "Spring":
        season =1
    else:
        season = 0
    bucketed.append(season)
    holiday = row_input[11]
    if holiday == "Holiday":
        holiday = 1
    else:
        holiday = 0
    bucketed.append(holiday)
    fd = row_input[12]
    if fd == "Yes":
        fd = 1
    else:
        fd =0
    bucketed.append(fd)
    return bucketed
def is_busy(row_input):
    row_input = bucket_single_row(row_input)
    features = list(df.columns)
    features.remove("Rented Bike Count")
    dt = DecisionTree(features = features)
    x_train = df.sample(frac=0.8)
    x_train = x_train.reset_index(drop=True)
    dt.train(x_train)
    x = dt.predict_row(row_input)
    if x == 0:
        print(x)
    else:
        print(x)
    return x


# ----------------------------------------------------------------------------------------------------------------------
df = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape')
bucketing(df)
#----------------------------------------------------------------------------------------------------------------------






