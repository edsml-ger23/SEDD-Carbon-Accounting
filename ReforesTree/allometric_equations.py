#allometric_equations.py

import numpy as np

def predict_cacao(diameter):
    a = 0.12079999999052336
    b = 1.9800000000506792
    return a * diameter ** b

def predict_guaba(diameter):
    a = 0.07762471189382622
    b = 2.639999998556892
    return a * diameter ** b

def predict_mango(diameter):
    a = 0.07762471165071548
    b = 2.640000000114163
    return a * diameter ** b

def predict_musacea(diameter):
    a = 0.03
    b = 2.13
    return a * diameter ** b

def predict_otra_variedad(diameter):
    a = 0.18231878714595798
    b = 2.153730277413256
    return a * diameter ** b

def calculate_AGB(diameter, species): 
    #regularize string of species
    species = species.lower()
    if species == 'cacao':
        return predict_cacao(diameter)
    elif species == 'guaba':
        return predict_guaba(diameter)
    elif species == 'mango':
        return predict_mango(diameter)
    elif species == 'musacea':
        return predict_musacea(diameter)
    elif species == 'otra variedad':
        return predict_otra_variedad(diameter)
    else:
        raise ValueError(f"Species {species} is not supported")
