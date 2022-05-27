#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import pickle
import zipfile
import pandas as pd

from flask import Flask, request, jsonify


# In[2]:


dias_semana = ['domingo', 'segunda-feira', 'terça-feira', 'quarta-feira', 'quinta-feira', 'sexta-feira', 'sábado']
with open('features.pkl', 'rb') as file:
    times = pickle.load(file)
    arenas = pickle.load(file)
    uf_estados = pickle.load(file)


# In[3]:


def get_resultado(n):
    if n==0:
        return 'Empate'
    elif n==1:
        return 'Vitória do Mandante'
    elif n==2:
        return 'Vitória do Visitante'
    
def convert_input(input_array):
    rodada = input_array[0]
    dia = dias_semana.index(input_array[1])
    
    check_new(times, input_array[2])
    mandante = times.index(input_array[2])
    check_new(times, input_array[2])
    visitante = times.index(input_array[3])
    
    check_new(arenas, input_array[4])
    arena = arenas.index(input_array[4])
    
    check_new(uf_estados, input_array[5])
    estado_mandante = uf_estados.index(input_array[5])
    check_new(uf_estados, input_array[6])
    estado_visitante = uf_estados.index(input_array[6])
    
    pontos_mandante = input_array[7]
    pontos_visitante = input_array[8]
    
    return [rodada, dia, mandante, visitante, arena, estado_mandante, estado_visitante, pontos_mandante, pontos_visitante]

def check_new(array, value):
    if value not in array:
        array.append(value)
        
    return array

def unzip_and_load_model():
    with zipfile.ZipFile('pfi_brasileirao.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
        
    model = joblib.load('pfi_brasileirao.joblib')
    
    return model


# In[4]:


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    data = convert_input(list(json_data.values()))
    proba = model.predict_proba([data])[0]
    predict = list(zip(model.classes_, proba))

    return jsonify({'empate': proba[0], 'vitoria mandante': proba[1], 'vitoria visitante': proba[2]})


# In[5]:


if __name__ == '__main__':
    model = unzip_and_load_model()
    app.run()


# In[ ]:




