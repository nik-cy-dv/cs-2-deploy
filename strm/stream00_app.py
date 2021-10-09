from __future__ import division
import streamlit as st
import pandas as pd
#from plot import plot_dispatch, plot_min_cost_dispatch
#from dispatch import dispatch_max_sc, dispatch_max_sc_grid_pf, dispatch_min_costs
#from analysis import print_analysis, print_min_cost_analysis
import time
import streamlit as st
import requests
import base64
import io
from PIL import Image
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model #, load_weights
from tensorflow.keras import Model as model
from pathlib import Path
#import keras




st.set_page_config(layout="wide")



# Title
st.title("Webapp for predicting insurance claim amount of vehicle")
st.markdown('A Web App by Nikhil ([@nikhil](https://www.linkedin.com/in/nikhil-sharma-421b60a6))')
st.markdown("")
st.markdown("Hey there! Welcome to my app. This app lets the user try out prediction" 
            "of insurance claim amount based on condition damaged or not. You can"
            " vary any of the parameters to see how they affect the performance. "
            "**Give it a go!**")

with st.beta_expander("Data Information"):
    st.markdown("prelimanary analysis data is taken from https://www.kaggle.com/infernape/fast-furious-and-insured")
st.markdown("")


pt = st.text_input('Enter Image URL',)


# Sidebar Content

imag = Image.open('strm/image/motorclaim.jpg')
st.sidebar.image(imag, width=280)
st.sidebar.write('#### Select an image of vehicle to upload or enter url of image on right side')
uploaded_file = st.sidebar.file_uploader('',
                                         type=['png', 'jpg', 'jpeg'],
                                         accept_multiple_files=False)


st.sidebar.write('#### Select insurance company')
Insurance_company = st.sidebar.selectbox("",
                                       ["BQ", 'A', 'AC', 'BC', 'DA', 'BB', 'C', 'O', 'B', 'AA', 'RE'])


st.sidebar.write('#### Select cost of vehicle')
Cost_of_vehicle = st.sidebar.slider(label="",
                         min_value=11000, max_value=54000)#, value=100, format='Week %d')



st.sidebar.write('#### Select minimum coverage offered for vehicle')
Min_coverage = st.sidebar.slider(label="",
                         min_value=277, max_value=1400)#, /stepvalue=0.5, format='Week %d')


#st.title("Date range")

min_date = datetime.datetime(2021,11,1)
max_date = datetime.date(2028,12,31)

st.sidebar.write('#### Pick a policy expiry date')
Expiry_date = st.sidebar.date_input("", min_value=min_date, max_value=max_date)



st.sidebar.write('#### Select maximum coverage offered for vehicle')
Max_coverage = st.sidebar.slider(label="",
                         min_value=2800, max_value=47000) #step=500


import pickle
import datetime
from PIL import Image
import cv2
import requests
#from google.colab.patches import cv2_imshow


luxury_seg = pickle.load(open('pickle/luxury_seg.pkl', 'rb'))
medium_seg = pickle.load(open('pickle/medium_seg.pkl','rb'))
cmp_cnt =  pickle.load(open('pickle/cmp_cnt.pkl','rb'))
ohe = pickle.load(open('pickle/ohe.pkl', 'rb'))
feature_labels = pickle.load(open('pickle/labels.pkl', 'rb'))
scaler = pickle.load(open('pickle/scaler.pkl', 'rb'))
gbdt = pickle.load(open('pickle/imgonly_gbr_model.pkl', 'rb'))
#path = 'model3_img_only.hdf5'#'blob/main/model3_deepl.h5'#'cs-2-deploy/blob/main/model3_deepl.h5'
#path = 'cs-2-deploy/blob/e719d60b01fa9e62498084a5f014a57b8d642910/model3_img_only.hdf5'


def feature_engg(data, col1, col2, col3, col4, col5):
  
  '''takes data : train/test dataframe,
     col1       : expiry_date,
     col2       : cost-of-vehicle,
     col3       : insurance company,
     col4       : max coverage
     col5       : min coverage
     returns    : newly computed feature in dataframe'''


  today = datetime.datetime(2021,9,12,0,0,0)
  #luxury_seg = np.percentile(data[col2], 75)
  #medium_seg = np.percentile(data[col2], 25)
  #cmp_cnt = data[col3].value_counts()


  data[col1]           = data[col1].apply(pd.to_datetime)
  data['year']         = data[col1].apply(lambda x : x.year)
  data['month']        = data[col1].apply(lambda x: x.month)
  data['month_day']    = data[col1].apply(lambda x: x.day)
  data['yr_day']       = data[col1].apply(lambda x: x.dayofyear)
  data['week_day']     = data[col1].apply(lambda x: x.weekday())
  data['week_no']      = data[col1].apply(lambda x: x.week)
  data['lux_seg']      = data[col2].apply(lambda x: 1 if x>luxury_seg else 0)
  data['med_seg']      = data[col2].apply(lambda x: 1 if (x<luxury_seg and x>medium_seg) else 0)
  data['budget_seg']   = data[col2].apply(lambda x: 1 if (x<medium_seg) else 0)
  data['age_of_insur'] = data[col1].apply(lambda x: round(abs((today-x).days)/365,2))

  md_age = np.median(data['age_of_insur'])

  data['cmpny_count']       = data[col3].apply(lambda x: cmp_cnt[x])
  data['range_of_coverage'] = data[col4]-data[col5]
  data['insuran_pd']        = data['age_of_insur'].apply(lambda x: 1 if x > md_age else 0)
  data['low_expire']        = data['age_of_insur'].apply(lambda x: 1 if x < 2 else 0)
  data['med_expire']        = data['age_of_insur'].apply(lambda x: 1 if (x > 2 and x<5) else 0)
  data['hig_expire']        = data['age_of_insur'].apply(lambda x: 1 if  x>5 else 0)
  data['cost_grt_20k']      = data[col4].apply(lambda x : 1 if x > 20000 else 0)

  return data

    
@st.cache(suppress_st_warning=True)
def model_pred(frame):
  
  p = './model3_deepl.h5'
  #if not os.path.exists(p):
  encoder_url = 'wget -O ./model3_deepl.h5 https://www.dropbox.com/home?preview=model3_deepl.h5'
  
  #cloud_model_location = "1BvZxcH_aO0udEecTXBhyVATBXDQz7YBU"
            
  #f_checkpoint = Path("model3_deepl.h5")

  #if not f_checkpoint.exists():
  #   with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
  #      from GD_download import download_file_from_google_drive
  #      download_file_from_google_drive(cloud_model_location, f_checkpoint)
            
            
  #with st.spinner('Downloading model weights'):
  #   os.system(encoder_url)
  #  else:
  #      print("Model 2 is here.")
    
  #best_model = tf.lite.TFLiteConverter.from_keras_model(p)
  #best_model  = model.load_weights('model3_deepl_wt.h5')
  best_model  = load_model('./model3_deepl.h5')
  if uploaded_file is not None:
    # User-selected image.
    content = Image.open(uploaded_file)
    st.image(content, caption='Original Image', use_column_width=True)
    t_image = content.resize((224, 224))
    t_image = preprocessing.image.img_to_array(t_image)
    t_image = t_image / 255.0                            #------------4
    t_image = np.expand_dims(t_image, axis=0)            #------------5
    prediction = best_model(t_image)
    frame['Condition'] = [1 if prediction[0][0]>0.5 else 0]
    

  if uploaded_file is None:
    content = requests.get(pt).content
    t_image = Image.open(BytesIO(content))
    st.image(t_image, caption='Original Image', use_column_width=True) 
    t_image = tf.image.decode_jpeg(content, channels=3)
    t_image = preprocessing.image.img_to_array(t_image) 
    t_image = tf.image.resize(t_image,[224,224])
    t_image = t_image / 255.0 
    t_image = np.expand_dims(t_image, axis=0)
    prediction = best_model(t_image)
    frame['Condition'] = [1 if prediction[0][0]>0.5 else 0]


  return frame

def ohe_transform_inter(arry, label, dataframe):

  '''takes arry : one-hot encoded columns,
          labels: names to be given to arry
        returns : final dataframe after concatating one hot columns with rest of features and droping some of them not usefull'''

  ohe_df = pd.DataFrame(arry, columns=label)   
  final = pd.concat([dataframe, ohe_df], axis=1)  
  final.drop(['Image_path','Insurance_company', 'Expiry_date'], axis=1, inplace=True)

  return final




#@st.cache(allow_output_mutation=True)
if (st.button('Predict claim amount')):
    data_frame = pd.DataFrame(list(zip([Image], [Insurance_company], [Cost_of_vehicle], [Min_coverage], [Expiry_date], [Max_coverage])) ,columns=['Image_path', 'Insurance_company', 'Cost_of_vehicle', 'Min_coverage', 'Expiry_date', 'Max_coverage'])
    #data_frame = pd.DataFrame([Image, Insurance_company, Cost_of_vehicle, Min_coverage, Expiry_date, Max_coverage] ,columns=['Image_path', 'Insurance_company', 'Cost_of_vehicle', 'Min_coverage', 'Expiry_date', 'Max_coverage'])
    #print(data_frame.head())
    fe_df = feature_engg(data_frame, 'Expiry_date', 'Cost_of_vehicle', 'Insurance_company', 'Max_coverage', 'Min_coverage')
    

    cond_frm = model_pred(fe_df)#, 'Image_path')#_path')
    test_arr1 = ohe.transform(cond_frm['Insurance_company'].values.reshape(-1,1)).toarray()
    final_test_1 = ohe_transform_inter(test_arr1, feature_labels, cond_frm)

    amount = round(gbdt.predict(final_test_1)[0], 3)
    st.write("Predicted claim amount ",     amount)
