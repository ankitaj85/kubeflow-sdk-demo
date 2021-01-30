import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from tensorflow.python.lib.io import file_io 
import tensorflow as tf
import config
import os

df = pd.read_csv(config.processed_data)

#### Get features ready to model! 
y = df.pop("cons_general").to_numpy()
y[y< 4] = 0
y[y>= 4] = 1

X = df.to_numpy()
X = preprocessing.scale(X) # Is standard
# Impute NaNs

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
X = imp.transform(X)


# Linear model
clf = LogisticRegression()
yhat = cross_val_predict(clf, X, y, cv=5)

acc = np.mean(yhat==y)
tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp + fn)

# Now print to file
with tf.io.gfile.GFile(config.store_artifacts + "/metrics.json", 'w') as outfile:
        json.dump({ "accuracy": acc, "specificity": specificity, "sensitivity":sensitivity}, outfile)

# Let's visualize within several slices of the dataset
score = yhat == y
score_int = [int(s) for s in score]
df['pred_accuracy'] = score_int

# Bar plot by region

sns.set_color_codes("dark")
ax = sns.barplot(x="region", y="pred_accuracy", data=df, palette = "Greens_d")
ax.set(xlabel="Region", ylabel = "Model accuracy")
plt.savefig("by_region.png",dpi=80)

from google.cloud import storage
storage_client = storage.Client()
bucket = storage_client.bucket(config.gs_bucket_name)
bucket.blob(str(config.version) + "/" + 'by_region.png').upload_from_filename('by_region.png', content_type='image/png')
print("Training Sucessfully Completed")


# Saving Model Artifacts like confusuion metrics

vocab = list(np.unique(y))
cm = confusion_matrix(y, yhat, labels=vocab)

data = []
for target_index, target_row in enumerate(cm):
    for predicted_index, count in enumerate(target_row):
        data.append((vocab[target_index], vocab[predicted_index], count))

df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
cm_file = os.path.join(config.store_artifacts, 'confusion_matrix.csv')
with file_io.FileIO(cm_file, 'w') as f:
    df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)

metadata = {
    'outputs' : [{
      'type': 'confusion_matrix',
      'format': 'csv',
      'schema': [
        {"name": "target", "type": "CATEGORY"},
        {"name": "predicted", "type": "CATEGORY"},
        {"name": "count", "type": "NUMBER"},
      ],
      'source': cm_file,
      # Convert vocab to string because for bealean values we want "True|False" to match csv data.
      'labels': list(map(str, vocab)),
    }
    ]
  }
with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
    json.dump(metadata, f)

metrics = {
    'metrics': [{
      'name': 'accuracy-score',
      'numberValue':  acc,
      'format': "PERCENTAGE",
    }, 
    {
      'name': 'specificity',
      'numberValue':  specificity,
      'format': "PERCENTAGE",
    }, 
    {
      'name': 'sensitivity',
      'numberValue':  sensitivity,
      'format': "PERCENTAGE",
    },]
  }
with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
    json.dump(metrics, f)