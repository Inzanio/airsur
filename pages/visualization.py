import streamlit as st
import pandas as pd
import numpy as np
import datetime

st.title("✨Visualisation des données des particules 📊")


col1, col2 , col3 = st.columns(3)





# visualisation de l'évolution de la qualité de l'air d'une particule ou de la temperature ou de l'humidité à une location précise
options = ["P0","P2","P1","humidity","temperature"]
st.write("👇")
features = st.pills("Particules", options=options,selection_mode="multi",default=options[0])

if features :
    df_features = pd.DataFrame()
    for feature in features :
        df_feature = pd.read_csv(f"data/aggregate_{feature}.csv")

        if df_features.empty:
            df_features = df_feature
        else:
            df_features = pd.merge(df_features, df_feature, on=['date', 'location'])

    #df_features
    df_max_value_features = df_features.groupby('location').max().reset_index()
    #df_max_value_features
    #     st.write(df_max_p0)

    df_features["date"] = pd.to_datetime(df_features["date"])


    # dates = [d for d in ]2024/2024/19
    dates = sorted(df_features["date"].to_list())
    col_debut , col_fin = st.columns(2)
    start_date = col_debut.date_input("🗓️Date de début", datetime.date(2024,1, 1), min_value=dates[0], max_value=dates[-1])
    end_date = col_fin.date_input("🗓️Date de Fin", dates[-1], min_value=start_date, max_value= dates[-1])

    start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
    end_date = datetime.datetime.combine(end_date, datetime.datetime.min.time())
    
    #st.write("You selected wavelengths between", start_date, "and", end_date)
    data = df_features[df_features["date"].between(start_date, end_date)]

    
    colgraph , colmap= st.columns(2)
    # # la map
    # # location et distributions des différents capteurs

    # #location_feature = df_p0

   
    df_locations = pd.read_csv("data/locations.csv")
    df_merge = pd.merge(df_max_value_features, df_locations, on='location')
    
    
    
    df_merge = df_merge.apply(lambda x: x.clip(upper=1000) if x.dtype.kind in 'bifc' else x)
    
    
    
    for feature in   features :
        st.write(f"📈Visualisation de {feature}")
        w = st.container(border=True)
        w.line_chart(data, x="date", y=feature)
        w.write("🌎Les Zones où les mesures on été prises📍")
        w.map(df_merge, size=feature)

