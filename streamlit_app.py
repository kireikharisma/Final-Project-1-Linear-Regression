import streamlit as st
import pickle
import numpy as np
import pandas as pd

#load model
transform_poly = pickle.load(open("poly_reg_3.pkl", "rb"))
encoding = pickle.load(open("encoding.pkl", "rb"))
model_fix = pickle.load(open("model_poly3.pkl", "rb"))
data = pd.read_parquet("data.parquet")

#load data
#archive = zipfile.ZipFile('Data.zip', 'r')
#xlfile = archive.open('Data.csv')
#data = pd.read_csv('Data.csv')


# judul web
st.title ('Prediksi Harga Transportasi Online')
st.markdown(
    "<p style='text-align: center;'>Made by <b><a href='https://www.linkedin.com/in/rosita-nurul-janatin-561145214/'>'Rosita Nurul Janatin</a></b> , <b><a href='https://www.linkedin.com/in/haikalefendi/'>'Haikal Efendi</a></b> & <b><a href='https://www.linkedin.com/in/ni-made-kirei-kharisma-handayani-90528b21a/'>Ni Made Kirei Kharisma Handayani</a></b></p>",
    unsafe_allow_html=True
)
st.image("https://asset-a.grid.id/crop/0x0:0x0/700x465/photo/2019/03/02/1743280914.jpeg")

#membagi kolom
col1, col2, col3 = st.columns(3)

with col1 : 
    global cab_type, name, source
    cab_type = st.selectbox ('Merk Transportasi Online', ('Lyft', 'Uber'))
    name = st.selectbox ('Jenis Mobil', list(np.sort(data['name'].unique())))
    source = st.selectbox ('Titil Awal', list(np.sort(data['source'].unique())))

with col2 :  
    global destination, distance, short_summary  
    destination = st.selectbox ('Tujuan Akhir', list(np.sort(data['destination'].unique())))
    short_summary = st.selectbox ('Cuaca', list(np.sort(data['short_summary'].unique())))
    distance = st.number_input ('Jarak Tempuh')

with col3 :   
    global windSpeed, visibility, surge_multiplier 
    windSpeed = st.number_input ('Kecepatan Angin')
    visibility = st.number_input ('Jarak Penglihatan')
    surge_multiplier = st.selectbox ('Kenaikan Harga Jika Penumpang Melonjak', list(np.sort(data['surge_multiplier'].unique())))

feature = [[cab_type,
            name,
            source,
            destination,
            short_summary,
            ]]

feature = pd.DataFrame(feature, columns=['cab_type', 'name', 'source', 'destination', 'short_summary'])

encode_feature = pd.DataFrame(encoding.transform(feature), columns=['cab_type', 'name', 'source', 'destination', 'short_summary'])

# membuat tombol untuk prediksi
if st.button('Harga Transportasi Online'):
    numerik = pd.DataFrame([[distance, windSpeed, visibility, surge_multiplier]])
    encode_feature = encode_feature.join(numerik)
    feature_fix = transform_poly.transform(encode_feature)
    price_prediction = model_fix.predict(feature_fix)
    st.success(f'Prediksi Harga Transportasi Online adalah ${price_prediction[0]:.2f} USD')

    