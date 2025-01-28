import streamlit as st


st.set_page_config(
    layout="wide",
    page_title="Qualité de l'air",
    page_icon="logo.png",
       
)
st.logo("logo.png")

home = st.Page("pages/home.py",title="Home", icon="🏠",default=True)
visual = st.Page("pages/visualization.py",title="Visualisation", icon="📈")
predict = st.Page("pages/predictions.py",title="Prediction", icon="🔮")
pages = [home,visual,predict]
# setting up app navigation
app = st.navigation(pages)
app.run()