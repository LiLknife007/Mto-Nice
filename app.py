
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
from datetime import datetime, timedelta

# Chargement du modèle et des scalers
model = tf.keras.models.load_model("model.keras")
scaler_X = joblib.load("scaler_X.save")
scaler_y = joblib.load("scaler_y.save")

# Fonction pour récupérer les prévisions météo 3h (et filtrer demain)
def get_6h_forecast(api_key):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q=Nice&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200 or "list" not in data:
        st.error("❌ Erreur dans la réponse de l'API. Vérifie ta clé API.")
        st.stop()

    forecast_list = data["list"]
    today = datetime.utcnow().date()
    tomorrow = today + timedelta(days=1)

    entries = []
    for item in forecast_list:
        timestamp = datetime.fromtimestamp(item["dt"])
        if timestamp.date() == tomorrow:
            entries.append({
                "time": timestamp.strftime("%Y-%m-%d %H:%M"),
                "temp": item["main"]["temp"],
                "rhum": item["main"]["humidity"]
            })

    # Garder 1 sur 2 pour passer à des prévisions toutes les 6 heures
    return pd.DataFrame(entries[::2])

# Interface utilisateur
st.title("🌤️ Prédiction météo IA - Nice (prévisions toutes les 6h)")
api_key = st.text_input("🔑 Entre ta clé API OpenWeatherMap", type="password")

if api_key:
    df = get_6h_forecast(api_key)
    st.write("📅 Données météo prévues demain (source API) :", df)

    # Prédictions IA
    X_scaled = scaler_X.transform(df[["temp", "rhum"]])
    pred_scaled = model.predict(X_scaled)
    pred = scaler_y.inverse_transform(pred_scaled)

    df["IA_temp_pred"] = pred[:, 0]
    df["IA_rhum_pred"] = pred[:, 1]

    st.subheader("📈 Prédictions IA toutes les 6h :")
    st.dataframe(df)

    st.subheader("✍️ Entrez vos vraies mesures pour comparer (optionnel)")
    for i, row in df.iterrows():
        df.loc[i, "real_temp"] = st.number_input(f"🌡️ Temp réelle à {row['time']} :", value=0.0, key=f"temp_{i}")
        df.loc[i, "real_rhum"] = st.number_input(f"💧 Humidité réelle à {row['time']} :", value=0.0, key=f"rhum_{i}")

    if st.button("📊 Comparer les prédictions IA avec les vraies mesures"):
        valid_rows = df[(df["real_temp"] > 0) & (df["real_rhum"] > 0)]

        if len(valid_rows) == 0:
            st.warning("Aucune vraie donnée saisie pour comparer.")
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error

            mse = mean_squared_error(valid_rows[["real_temp", "real_rhum"]],
                                     valid_rows[["IA_temp_pred", "IA_rhum_pred"]])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(valid_rows[["real_temp", "real_rhum"]],
                                      valid_rows[["IA_temp_pred", "IA_rhum_pred"]])

            st.success(f"📉 MSE : {mse:.2f}")
            st.success(f"📉 RMSE : {rmse:.2f}")
            st.success(f"📉 MAE : {mae:.2f}")
