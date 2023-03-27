import re

import pandas as pd
import sklearn.model_selection
import sklearn.pipeline
import sklearn.feature_extraction
import sklearn.neural_network
import sklearn.metrics


def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+', '__URL__', text)
    text = re.sub(r'@\S+', '__AT_USER__', text)
    text = re.sub(r'#(\S+)', r'\1', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df=pd.read_csv("fb_sentiment.csv")
df['FBPost'] = df['FBPost'].apply(preprocess_text)
df = df.sample(frac = 1)
train_data=list(df.loc[:,['FBPost']].values)
train_target=list(df.loc[:,['Label']].values)
trd=[]
for arr in train_data:
    trd.append(arr[0])
trt=[]
for arr in train_target:
    trt.append(arr[0])
train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
trd, trt, test_size=0.2, random_state=42)

model = sklearn.pipeline.Pipeline(
    [('preproc_char', sklearn.feature_extraction.text.TfidfVectorizer(analyzer='word', ngram_range=(1, 9),
                                                                        sublinear_tf=True, lowercase=True)),
        ('model', sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(600,200,60,20),verbose=True,learning_rate_init=0.0005,tol=0.00001))])
model.fit(train_data, train_target)