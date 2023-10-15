import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/content/own_dataset.csv')
df.head()
# function to remove pucntuation and clean string
def remove_pun(text):
  for punctuation in string.punctuation:
    text = text.replace(punctuation,"")
  text = text.lower()
  return(text)
# storing cleaned text
df['Text']=df['Text'].apply(remove_pun)
from sklearn.model_selection import train_test_split
# text in X as df[0] , lanuage in Y as df[1]
X = df.iloc[:,0]
Y = df.iloc[:,1]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25)
# let us say we have a document
# d1 = I am Raj
# d2 = I am Vishnu
# this can be featurised by unigram or bygram or trigram

#             feature 1                 feature 2             feature 3
# unigram       I                         am                    Raj
# bigram        I am                      am Raj
# trigram       I am Raj

#               [I    am    Raj]    Vishnu]
#         d1 =  [1    1       1]         0]
#         d2 =  [1    1       0]         1]

# It is not very good as frequency becomes a problem as repatitve words so we use TF.IDF
# TF is term freqency with in a single document or text.
# IDF is Inverse document Frequency

# This is used for vecatorization to create and focous on values which appear less in document
from sklearn import feature_extraction
vec = feature_extraction.text.TfidfVectorizer(ngram_range=(1,2),analyzer='char')
from sklearn import pipeline
from sklearn import linear_model
model_pipe = pipeline.Pipeline([('vec',vec),('clf',linear_model.LogisticRegression())])
model_pipe.fit(X_train,Y_train)
predict_val = model_pipe.predict(X_test)
from sklearn import metrics
#metrics.accuracy_score(Y_test,predict_val)*100
model_pipe.predict(['हिन्दी कहानी एक रचना है, जो जीवन के किसी एक अंग या मनोभाव को प्रदर्शित करती है । कहानी सुनने, पढ़ने और लिखने की एक लम्बी परम्परा हर देश में रही है; क्योंकि यह मन को रमाती है और सबके लिए मनोरंजक होती है। आज हर उम्र का व्यक्ति कहानी सुनना या पढ़ना चाहता है यही कारण है कि कहानी का महत्त्व दिन-दिन बढ़ता जा रहा है। हर कहानी का अपना एक अलग उद्देश्य होता है कुछ कहानियाँ हमे कोई सिख प्रदान करती है, कुछ हमे मनोरंजन कराती है, कुछ जीवन के संघर्ष के बारे में बताती है तो कुछ हमे धार्मिक बातों की ओर ले जाती'])
