import numpy as np
import pandas as pd
def divide_data(x_data, fkey, fval):
    x_right = pd.DataFrame([], columns=x_data.columns)
    x_left = pd.DataFrame([], columns=x_data.columns)
    for ix in range(x_data.shape[0]):
        # Retrieve the current value for the fkey column lets call it val
        try:
            #TODO
            val = x_data.iloc[ix][fkey]
            val_row = x_data.iloc[ix,:]
        except:
            #TODO
            break
        if val > fval:
            x_right = x_right.append(val_row)
        else:
            x_left = x_left.append(val_row)
    series = pd.Series()
    series['0'] = x_left
    series['1'] = x_right
    return series

def entropy(data, col):
    #total number of data
    S = col.shape[0]
    #number of class of data
    values = col.unique()
    H = 0
    part1 = 0
    part2 = 0
    split_info = 0
    for i in values:
        #number of data of class i
        df1 = data.loc[col == i]
        Sv = df1.shape[0]
        #splitInfor
        split_info += -(Sv/S)*np.log2(Sv/S)
        #number of pos
        pos = df1[df1.Survived==1].shape[0]
        #number of neg
        neg = df1[df1.Survived==0].shape[0]
        if(pos == 0):
            part1 = 0
        elif(pos!=0):
            part1 = -(pos/Sv)*np.log2(pos/Sv)
        if(neg == 0):
            part2 = 0
        elif(neg!=0):
            part2 = -(neg/Sv)*np.log2(neg/Sv)
        
        H+=(Sv/S)*(part1 + part2)
        
    return H,split_info
        
def information_gain(xdata, fkey, fval):
    #TODO
    total = xdata.shape[0]
    #initial pos
    pos1 = xdata[xdata.Survived==1].shape[0]
    #initial neg
    neg1 = xdata[xdata.Survived==0].shape[0]
    Intial = ((-pos1/total)*np.log2(pos1/total)-(neg1/total)*np.log2(neg1/total))
    Entropy,Split_infor = entropy(xdata,xdata[fkey])
    return (Intial - Entropy)/Split_infor

class DecisionTree:
    def __init__(self, depth=0, max_depth=5):
        self.left = None
        self.right = None
        self.fkey = None
        self.fval = None
        self.max_depth = max_depth
        self.depth = depth
        self.target = None
    
    def train(self, train,d,dd):
        Tree = dd
        input_cols = train.columns.drop('Survived').tolist()
        out_cols = ['Survived']
        col_dict = {'Pclass':0, 'Sex':1, 'Age':2, 'SibSp':3, 'Parch':4, 'Fare':5}
        X_train = train[input_cols]
        y_train = train[out_cols]
        self.depth = d
        if(self.depth > self.max_depth):
            zero = y_train[y_train.Survived==0].shape[0]
            one = y_train[y_train.Survived==1].shape[0]
            if(zero > one):
                return 0
            else:
                return 1
        gains = {}
        for fx in X_train.columns:
            g = information_gain(train, fx, train[fx].mean())
            gains[fx] = g
        bestFeature = max(gains, key=gains.get)
        split_series = divide_data(train, bestFeature, train[bestFeature].mean())
        Tree = {bestFeature:{}}
        
        for j in range(len(split_series)):
            split_data = split_series.iloc[j]
            if(j == 0):
                value = '<= '+str(train[bestFeature].mean())
            else:
                value = '>'+str(train[bestFeature].mean())
            self.depth+=1
            split_data = split_data.drop(bestFeature, axis = 1) 
            
            Tree[bestFeature][value] = self.train(split_data,self.depth,Tree) 
        return Tree
    
    def predict(self,rules1,data1):
        if(isinstance(rules1,int)):
            return rules1 

        for i in rules1.keys():
            if(i in data1):
                val = data1[i]
                for j in rules1[i].keys():
                    jj = str(val) + j
                    if(eval(jj)):
                        result = self.predict(rules1[i][j],data1)
        return result