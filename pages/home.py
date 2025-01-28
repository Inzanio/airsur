import streamlit as st

st.title("🎉Bienvenue sur :blue[Airsur]")
st.write(":blue[Airsur], l'application de prédiction de la qualité de l'air à Nairobi.")

cols = st.columns(2,vertical_alignment="center")

options = {
    "Vizulisation" : {
        "page" : "pages/visualization.py",
        "description" : "Visualiser les données collectées sur la qualités de l'air à Nairobi",
        "label" : ":blue[Visualiser]",
        "icon":  "📈"
    },
    "Prédiction de la qualité de l'air" : {
        "page":"pages/predictions.py",
        "description":"Appreciez nos prédictions sur la concentration des particules toxiques dans l'air",
        "label": ":blue[Apprecier !]",
        "icon":"👀"
    },
    # "Sélection des capteurs" : {
    #     "page" : "pages/sentence_similarity.py",
    #     "description" : "Voulez vous comparer la similiratié entre des phrases ?",
    #     "label" : ":blue[Comparer !]",
    #     "icon": "🧮"
    # },
    # "Lire du texte" : {
    #     "page":"pages/text_to_speech.py",
    #     "description":"Voulez vous que nous vous lisions du texte ?",
    #     "label": ":blue[Lire !]",
    #     "icon" : "🎧"
    # },
}

for i, items in enumerate (options.items()) :
    with cols[i % 2]:  # Répartir les expanders sur les colonnes
        w = st.container(border=True,height = 225)
        w.subheader(items[0])
        w.write(items[1]["description"])
        w.page_link(items[1]["page"],label=items[1]["label"], icon=items[1]["icon"])