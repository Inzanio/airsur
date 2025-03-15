import streamlit as st
import datetime
import pandas as pd
import numpy as np

from keras.models import load_model
from keras.metrics import MeanAbsoluteError
import tensorflow as tf

if "temperature" not in st.session_state :
    st.session_state.temperature = 0
if "humidity" not in st.session_state :
    st.session_state.humidity = 0
if "df_h" not in st.session_state :
    st.session_state.df_h = None

def r2_keras(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

def correct_value (value, humidity):
    return 0.52 * value - 0.085 * humidity + 5.71

st.title("🚀Prédictions de la concentration des particules🌟")
#model_options = ["Prophète","LSTM","ARIMA"]
#model_name = st.pills("Choix du modèle",model_options, default=model_options[0])
features= ["humidity","temperature","P2","P1","P0"]
features_map = {
    "P0": ":blue[P0] Particules de diamètre inférieur à 1 µm",
    "P2": ":blue[P2] Particules de diamètre inférieur à 2,5 µm",
    "P1":":blue[P1] Particules de diamètre inférieur à 10 µm",
    "humidity":"Humidité",
    "temperature":""
}
cols_t_h = st.columns(2)
cols = st.columns(3)

date_prediction = st.date_input("📅Date de Prédiction",max_value= datetime.date(2024,3,20) + datetime.timedelta(days=7),value=datetime.date(2024,3,20))

for i, feature in enumerate (features) :
    if (feature not in [features[0],features[1]]):
        st.write(f":blue[{features_map[feature]}]")
    df = pd.read_csv(f'data/{feature}.csv')
    df= df.groupby('timestamp').max().reset_index()

    df = df[['timestamp', feature]]
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Renommer les colonnes pour utiliser Prophet


    # if model_name == model_options[0]:
    
    #     from prophet.serialize import model_from_json

    #     # Charger le modèle à partir d'un fichier JSON
    #     with open(f'models/modele_prophet_{feature}.json', 'r') as fichier:
    #         model = model_from_json(fichier.read())

    #if model_name == model_options[1] :
    # Importer les bibliothèques nécessaires
    
    # Charger le modèle à partir du fichier
    model = load_model(f'models/modele_lstm_{feature}.h5')
        
        #st.write(model)

    # if model_name == model_options[2] :
    #     # Importer les bibliothèques nécessaires
    #     import pickle

    #     # Charger le modèle à partir du fichier pickle
    #     with open(f'models/modele_arima_{feature}.pkl', 'rb') as fichier:
    #         model = pickle.load(fichier)
        
        #st.write(model)

    

    # max_pm2 = 
    # col1.metric("PM 2.5", "55.0 µg", "-5.0 µg",delta_color='inverse')
    # col2.metric("TEMPERATURE", "34.70 ℃", "-2.43 ℃")
    # col3.metric("HUMIDITY", "99.90% RH", "4% RH")

    


    #date_prediction

    # if model_name == model_options[0] :
    #     df.columns = ['ds', 'y']
    #     # Faire des prévisions
    #     #st.write(df['ds'].max())
    #     #p = (date_prediction - datetime.date(df['ds'].max().year,df['ds'].max().month,df['ds'].max().day)).days + 1
    #     #p = (date_prediction - df['ds'].max().date()).days + 1
    #     p = int((date_prediction - df['ds'].max().date()).total_seconds() / 3600)
    #     #st.write(p)
    #     future = model.make_future_dataframe(periods=p ) # + 304
    #     #st.write(future)
    #     forecast = model.predict(future)

    #     # Afficher les prévisions
    #     #st.write(forecast)
    #     forecast = pd.merge(forecast,df, on="ds",how="left")

    #     st.line_chart(forecast.rename(columns={'y': 'Concentration Réelles', 'yhat': 'Concentration Prédites'}),x="ds", y=["Concentration Réelles","Concentration Prédites"])
    #     last = forecast["yhat"].iloc[-1] 
    #     #st.write(last)
    #     before_last = forecast["yhat"].iloc[-2] 
    #     #st.write(last)
    #     cols[i].metric(f"{feature}", f"{round(last,1)} µg", f"{round(last-before_last,1)} µg",delta_color='inverse')
        
    #if model_name == model_options[1] :
    #st.write("En cours d'implémentation")
    df.set_index('timestamp', inplace=True)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    forecast_period = int((date_prediction - df.index.max().date()).total_seconds() / 3600) #(date_prediction - df.index.max().date()).days + 1
    #st.write(forecast_period)
    
    forecast = []
    
    seq_length = 48 
    # Use the last sequence from the test data to make predictions
    last_sequence = scaled_data[-seq_length:]
    #st.write(last_sequence)
    
    dates_forecast = pd.date_range(start=df.index.max(), periods=forecast_period,freq='H')
    
    for _ in range(forecast_period):
        # Reshape the sequence to match the input shape of the model
        current_sequence = last_sequence.reshape(1, seq_length, 1)
        # Predict the next value
        next_prediction = model.predict(current_sequence)[0][0]
        # if (feature in ["P2","P1","P0"]) :
      
        #     next_prediction = scaler.transform ([[correct_value(next_prediction,st.session_state.df_h["yhat"].iloc[-forecast_period+_])]])
        # Append the prediction to the forecast list
        forecast.append(next_prediction)
        # Update the last sequence by removing the first element and appending the predicted value
        last_sequence = np.append(last_sequence[1:], next_prediction)

    # Inverse transform the forecasted values
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    
    df_forecast = pd.DataFrame({'timestamp': dates_forecast, 'yhat': forecast.flatten()})
    forecast = df_forecast.copy()  #pd.concat([df.reset_index(),df_forecast],ignore_index=True)
    forecast["Seuil Attention"] = 30
    forecast["Seuil Danger"] = 120
    #st.write(forecast)
    #st.line_chart(forecast.rename(columns={feature: 'Concentration Réelles', 'yhat': 'Concentration Prédites'}),x='timestamp', y=["Concentration Réelles","Concentration Prédites"])
    if (feature not in [features[0],features[1]]):
        st.line_chart(forecast.rename(columns={'yhat': 'Concentration Prédites'}),x='timestamp', y=["Concentration Prédites","Seuil Attention","Seuil Danger"],color=["#2893d1","#a89932","#cc182a"])
    
    last = forecast["yhat"].iloc[-1] 
    #st.write(last)
    before_last = forecast["yhat"].iloc[-2] 
    #st.write(last)
    # st.write(f"{i} ,, {feature}")
    # st.write(feature == features[0])
    if (feature == features[0]):
        
        cols_t_h[1].metric(f"{feature}", f"{format(last, ".2f")} % RH")
       
        st.session_state.humidity = round(last,1)
        st.session_state.df_h = forecast
    elif (feature == features[1]):
        cols_t_h[0].metric(f"{feature}", f"{format(last, ".2f")} ℃") 
        st.session_state.temperature = format(last, ".2f")
        
    else :
        cols[i-2].metric(f"{feature}", f"{format(last, ".2f")} µg/m3") #, f"{round(last-before_last,1)} µg",delta_color='inverse'
    # if model_name == model_options[2] :
        
    #     forecast_period = (date_prediction - df.date.max().date()).days + 1
    #     forecast = model.forecast(steps=forecast_period)
        
    #     dates_forecast = pd.date_range(start=df.date.max(), periods=forecast_period)
    #     df_forecast = pd.DataFrame({'timestamp': dates_forecast, 'yhat': forecast})
    #     forecast = pd.concat([df,df_forecast],ignore_index=True)
    #     #st.write(forecast)
    #     st.line_chart(forecast.rename(columns={feature: 'Concentration Réelles', 'yhat': 'Concentration Prédites'}),x='timestamp', y=["Concentration Réelles","Concentration Prédites"])
        
    #     last = forecast["yhat"].iloc[-1] 
    #     #st.write(last)
    #     before_last = forecast["yhat"].iloc[-2] 
    #     #st.write(last)
    #     cols[i].metric(f"{feature}", f"{round(last,1)} µg", f"{round(last-before_last,1)} µg",delta_color='inverse')