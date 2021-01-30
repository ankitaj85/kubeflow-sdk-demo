import os
import wget
import tensorflow as tf
# data from https://www.sciencedirect.com/science/article/pii/S2352340920303048
import config
# Download the zipped dataset
url = 'https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/yshdbyj6zy-1.zip'
zip_name = "data.zip"
wget.download(url, zip_name)

# Unzip it and standardize the .csv filename
import zipfile
with zipfile.ZipFile(zip_name,"r") as zip_ref:
    zip_ref.filelist[0].filename = 'data_raw.csv'
    zip_ref.extract(zip_ref.filelist[0])
    
#import dask.dataframe as dd
#df = dd.read_csv('data_raw.csv') # nice!
#df.to_csv(config.data_path)
# df is now Dask dataframe, ready for distributed processing
# If you want to have the pandas version, simply:
#  df_pd = df.compute()
# upload file to google storage
from google.cloud import storage
storage_client = storage.Client()
bucket = storage_client.bucket(config.gs_bucket_name)
bucket.blob('data/data_raw.csv').upload_from_filename('data_raw.csv', content_type='text/csv')
print("Data DownLoaded Sucessfully")