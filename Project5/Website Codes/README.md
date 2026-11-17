# Project 05 – Titanic Survival Predictor

## Important: add your trained model

This ZIP contains the Streamlit website files. It does not contain
`model.joblib`, because that file was created in your Google Colab
session and is not available in this chat.

Copy your downloaded `model.joblib` into this folder, beside `app.py`.

The folder must contain:
- app.py
- model.joblib
- requirements.txt

## Run the website

Open a terminal in this folder and run:

    pip install -r requirements.txt
    streamlit run app.py

The app expects the model features in this order:
`pclass`, `sex`, `age`, `fare`

Gender encoding must match the training code:
female = 1, male = 0.
