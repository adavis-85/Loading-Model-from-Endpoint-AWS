#!/usr/bin/env python
# coding: utf-8

# # Importing Necessary Libraries 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc, accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score,classification_report
import seaborn as sns

import os
import io
import boto3
import json
import csv
from io import StringIO
import sagemaker


# # Load Endpoint and Key for Input

# In[2]:


sagemaker = boto3.client('sagemaker')
ENDPOINT_NAME = 'xgb-linsearch***********'
runtime= boto3.client('runtime.sagemaker')
bucket = '**********'
s3 = boto3.client('s3')
key = 'test_set.csv'


# # Reading Input and Gathering Predictions into Csv

# In[3]:


response = s3.get_object(Bucket=bucket, Key=key)
content = response['Body'].read().decode('utf-8')
results = []
for line in  content.splitlines():
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                            ContentType='text/csv',
                                            Body=line)
    result = json.loads(response['Body'].read().decode())
    results.append(result)
    i = 0
multiLine = ""
for item in results:
    if (i > 0):
        multiLine = multiLine + '\n'
    multiLine = multiLine + str(item)
    i+=1

file_name = "predictions.csv"
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket, file_name).put(Body=multiLine)


# # Reading in Predictions 

# In[4]:


object_key = 'predictions.csv'
csv_obj = s3.get_object(Bucket=bucket, Key=object_key)
body = csv_obj['Body']
csv_string = body.read().decode('utf-8')

predictions_array = pd.read_csv(StringIO(csv_string))


# # Reading in Target to Test Accuracy

# In[5]:


object_key = 'test_target.csv'
csv_obj = s3.get_object(Bucket=bucket, Key=object_key)
body = csv_obj['Body']
csv_string = body.read().decode('utf-8')

test = pd.read_csv(StringIO(csv_string))


# # Visualizing Accuracy of Model Endpoint

# In[6]:


cm = metrics.confusion_matrix(test ,np.round(predictions_array))
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[7]:


print(classification_report(test , np.round(predictions_array)))


# # Cleanup: Deleting Endpoint and Contents of Bucket

# In[8]:


# Deleting Endpoint
sagemaker.delete_endpoint(EndpointName=ENDPOINT_NAME)


# In[9]:


bucket_to_delete = boto3.resource('s3').Bucket(bucket)
bucket_to_delete.objects.all().delete()

