import streamlit as st
import datetime
import pandas as pd
import numpy as np

st.title("🚀Prédictions de la concentration des particules🌟")
model_options = ["Prophète","LSTM","ARIMA"]
model_name = st.pills("Choix du modèle",model_options, default=model_options[0])
features= ["P2","P1","P0"]
features_map = {
    "P0": ":blue[P0] Particules de diamètre inférieur à 1 µm",
    "P2": ":blue[P2] Particules de diamètre inférieur à 2,5 µm",
    "P1":":blue[P1] Particules de diamètre inférieur à 10 µm"
}

cols = st.columns(3)

date_prediction = st.date_input("📅Date de Prédiction",max_value= datetime.date.today() + datetime.timedelta(days=30))

for i, feature in enumerate (features) :
    st.write(f":blue[{features_map[feature]}]")
    df = pd.read_csv(f'data/aggregate_{feature}.csv')
    df= df.groupby("date").max().reset_index()

    df = df[['date', feature]]
    df['date'] = pd.to_datetime(df['date'])

    # Renommer les colonnes pour utiliser Prophet


    if model_name == model_options[0]:
    
        from prophet.serialize import model_from_json

        # Charger le modèle à partir d'un fichier JSON
        with open(f'models/modele_prophet_{feature}.json', 'r') as fichier:
            model = model_from_json(fichier.read())

    if model_name == model_options[1] :
        # Importer les bibliothèques nécessaires
        from keras.models import load_model

        # Charger le modèle à partir du fichier
        model = load_model(f'models/modele_lstm_{feature}.h5')
        
        #st.write(model)

    if model_name == model_options[2] :
        # Importer les bibliothèques nécessaires
        import pickle

        # Charger le modèle à partir du fichier pickle
        with open(f'models/modele_arima_{feature}.pkl', 'rb') as fichier:
            model = pickle.load(fichier)
        
        #st.write(model)

    

    # max_pm2 = 
    # col1.metric("PM 2.5", "55.0 µg", "-5.0 µg",delta_color='inverse')
    # col2.metric("TEMPERATURE", "34.70 ℃", "-2.43 ℃")
    # col3.metric("HUMIDITY", "99.90% RH", "4% RH")

    


    #date_prediction

    if model_name == model_options[0] :
        df.columns = ['ds', 'y']
        # Faire des prévisions
        #st.write(df['ds'].max())
        #p = (date_prediction - datetime.date(df['ds'].max().year,df['ds'].max().month,df['ds'].max().day)).days + 1
        p = (date_prediction - df['ds'].max().date()).days + 1
        #st.write(p)
        future = model.make_future_dataframe(periods=p+304 )
        #st.write(future)
        forecast = model.predict(future)

        # Afficher les prévisions
        #st.write(forecast)
        forecast = pd.merge(forecast,df, on="ds",how="left")

        st.line_chart(forecast.rename(columns={'y': 'Concentration Réelles', 'yhat': 'Concentration Prédites'}),x="ds", y=["Concentration Réelles","Concentration Prédites"])
        last = forecast["yhat"].iloc[-1] 
        #st.write(last)
        before_last = forecast["yhat"].iloc[-2] 
        #st.write(last)
        cols[i].metric(f"{feature}", f"{round(last,1)} µg", f"{round(last-before_last,1)} µg",delta_color='inverse')
        
    if model_name == model_options[1] :
        #st.write("En cours d'implémentation")
        df.set_index('date', inplace=True)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        
        forecast_period =  (date_prediction - df.index.max().date()).days + 1
        st.write(forecast_period)
        
        forecast = []
        
        seq_length = 30 
        # Use the last sequence from the test data to make predictions
        last_sequence = scaled_data[-30:]
        
        dates_forecast = pd.date_range(start=df.index.max(), periods=forecast_period+1)
        
        for _ in range(forecast_period):
            # Reshape the sequence to match the input shape of the model
            current_sequence = last_sequence.reshape(1, seq_length, 1)
            # Predict the next value
            next_prediction = model.predict(current_sequence)[0][0]
            # Append the prediction to the forecast list
            forecast.append(next_prediction)
            # Update the last sequence by removing the first element and appending the predicted value
            last_sequence = np.append(last_sequence[1:], next_prediction)

        # Inverse transform the forecasted values
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
        
        df_forecast = pd.DataFrame({'date': dates_forecast, 'yhat': forecast.flatten()})
        forecast = pd.concat([df.reset_index(),df_forecast],ignore_index=True)
        #st.write(forecast)
        st.line_chart(forecast.rename(columns={feature: 'Concentration Réelles', 'yhat': 'Concentration Prédites'}),x="date", y=["Concentration Réelles","Concentration Prédites"])
        
        last = forecast["yhat"].iloc[-1] 
        #st.write(last)
        before_last = forecast["yhat"].iloc[-2] 
        #st.write(last)
        cols[i].metric(f"{feature}", f"{round(last,1)} µg", f"{round(last-before_last,1)} µg",delta_color='inverse')
    if model_name == model_options[2] :
        
        forecast_period = (date_prediction - df.date.max().date()).days + 1
        forecast = model.forecast(steps=forecast_period)
        
        dates_forecast = pd.date_range(start=df.date.max(), periods=forecast_period)
        df_forecast = pd.DataFrame({'date': dates_forecast, 'yhat': forecast})
        forecast = pd.concat([df,df_forecast],ignore_index=True)
        #st.write(forecast)
        st.line_chart(forecast.rename(columns={feature: 'Concentration Réelles', 'yhat': 'Concentration Prédites'}),x="date", y=["Concentration Réelles","Concentration Prédites"])
        
        last = forecast["yhat"].iloc[-1] 
        #st.write(last)
        before_last = forecast["yhat"].iloc[-2] 
        #st.write(last)
        cols[i].metric(f"{feature}", f"{round(last,1)} µg", f"{round(last-before_last,1)} µg",delta_color='inverse')