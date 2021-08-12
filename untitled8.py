import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#%%
#Declare data frames for analysis
nsdq    = pd.read_csv("NQ=F (2).csv")
spy     = pd.read_csv("SPY (1).csv")
combi   = pd.merge(nsdq, spy, on = 'Date')

#%% 
def norm_fun(x): 
    
    high    = x[0]
    low     = x[0]
    new_x = []

    for n in range(0, len(x)):
        if ( x[n] > high):
            high = x[n]
        if ( x[n] < low):
            low = x[n]
            
    for n in range(0, len(x)):
        new_x.append((x[n] -low) - (high - low))

    return new_x




#%%
combi["dif_x"] = combi["Open_x"] - combi["Close_x"]
combi["dif_y"] = combi["Open_y"] - combi["Close_y"]

combi = combi.dropna()

tot_labels    = []
nsdq_train   = []
spy_train = []

sum_ = 0 

for x in range (1, len(combi)):
    # if(combi.iloc[x]["Open_x"] - combi.iloc[x]["Close_x"] >=
    #     combi.iloc[x]["Open_y"] - combi.iloc[x]["Close_y"]):
    #     tot_labels.append([1])
    if(combi.iloc[x]["Open_x"] > combi.iloc[x]["Close_x"] ):
        tot_labels.append([1])
        sum_ = sum_ +1
    else:
        tot_labels.append([0]) 
        sum_ = sum_ -1
    temp_list_x =norm_fun( [combi.iloc[x-1]["Open_x"], combi.iloc[x-1]["Close_x"], 
                            combi.iloc[x-1]["High_x"], combi.iloc[x-1]["Low_x"], 
                            combi.iloc[x]["Open_x"]])
    temp_list_y = norm_fun([combi.iloc[x-1]["Open_y"], combi.iloc[x-1]["Close_y"], 
                            combi.iloc[x-1]["High_y"], combi.iloc[x-1]["Low_y"], 
                            combi.iloc[x]["Open_y"]] )

    
    nsdq_train.append( temp_list_x )
    spy_train.append( temp_list_y )


nsdq_train = tf.keras.utils.normalize(nsdq_train, axis =1)
spy_train = tf.keras.utils.normalize(spy_train, axis=1)

tot_train = np.array(np.concatenate((nsdq_train,spy_train), axis = 1))



tot_labels = np.array(tot_labels)
encoder = LabelEncoder()
encoder.fit(tot_labels)
encoded_Y = encoder.transform(tot_labels)


x_train, x_test, y_train, y_test = train_test_split(tot_train, encoded_Y,
                                   test_size= 0.13, 
                                   random_state = 1)


#%%

new_model = tf.keras.models.load_model("test1.model")
predictions = new_model.predict(x_test)

score = 0
for n in range( 0, len(predictions)):
    if (y_test[n] == 1 and predictions[n] >0.5):
        score = score +1
        
    if(y_test[n] == 0 and predictions[n] <0.5):
        score = score +1


ratio  = score / len(predictions)
print(score)
print(ratio)