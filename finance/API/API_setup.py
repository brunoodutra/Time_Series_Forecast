
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import joblib
import os
import json
import pandas as pd
app = FastAPI()


# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens (ajuste conforme necessário)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos HTTP (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Permite todos os cabeçalhos
)

# function to get the last recomendation
def get_last_recommendation(model_name: str, crypto: str):
    file_path = f'./../AI/Classification/Real_Time_Inference/Recommendations/{model_name}_{crypto}_recommendation.csv'
    try:
        # Specify the delimiter (in this case, a comma)
        df = pd.read_csv(file_path, sep=',', header=0)
        last_recommendation = df.iloc[-1]
        return {
            'Date': last_recommendation['Date'],
            'Time': last_recommendation['Time'],
            'recommendation': last_recommendation['recommendation'],
            'percentage': last_recommendation['percentage'],
            'Price': last_recommendation['Price'],
        }
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
# function to get a specific date and time recomendation
def get_especific_recommendation(model_name: str, crypto: str, date: str, time: str):
    file_path = f'./../AI/Classification/Real_Time_Inference/Recommendations/{model_name}_{crypto}_recommendation.csv'
    try:
        # Specify the delimiter (in this case, a comma)
        df = pd.read_csv(file_path, sep=',', header=0)
        specific_recommendation = df[(df['Date'] == date) & (df['Time'] == time)]
        if not specific_recommendation.empty:
            row = specific_recommendation.iloc[0]
            return {
                'Date': row['Date'],
                'Time': row['Time'],
                'recommendation': row['recommendation'],
                'percentage': row['percentage'],
                'Price': row['Price'],
            }
        else:
            return None
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_recommendation_history(model_name: str, crypto: str, last_n: int = 200):
    file_path = f'./../AI/Classification/Real_Time_Inference/Recommendations/{model_name}_{crypto}_recommendation.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        if len(df) < last_n:
            last_n = len(df)
            
        df = df.iloc[-last_n:]  # Get the last n rows
        
        history = []
        for index, row in df.iterrows():
            history.append({
                'Date': row['Date'],
                'Time': row['Time'],
                'recommendation': row['recommendation'],
                'percentage': row['percentage'],
                'Price': row['Price'],
            })
        return history
    else:
        return None

def get_last_target_stop(model_name: str, crypto: str, profile: str):
    file_path = f'./../AI/Classification/Real_Time_Inference/Recommendations/{model_name}_{crypto}_recommendation.csv'
    try:
        # Carrega o arquivo CSV
        df = pd.read_csv(file_path, sep=',', header=0)
        
        # Filtra a última linha
        last_row = df.iloc[-1]
        
        # Converte a string JSON para um objeto Python
        signal_gains = json.loads(last_row['SignalGains'])
        
        # Obtem os valores de target e stop loss para o perfil especificado
        target = signal_gains[profile]['target']
        stop_loss = signal_gains[profile]['stop_loss']
        
        return {
            'target': target,
            'stop_loss': stop_loss
        }
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# API para obter os valores de target e stop loss
@app.get("/last_target_stop")
async def last_target_stop_api(model_name: str, crypto: str, profile: str):
    last_target_stop = get_last_target_stop(model_name, crypto, profile)
    if last_target_stop is not None:
        return last_target_stop
    else:
        return {"error": "No recommendations found"}
        
# API to get the last recomendation
@app.get("/last_recommendation")
async def last_recommendation_api(model_name: str, crypto: str):
    last_recommendation = get_last_recommendation(model_name, crypto)
    if last_recommendation is not None:
        return last_recommendation
    else:
        return {"error": "No recommendations found"}

# API to get the history 
@app.get("/recommendation_history")
async def last_recommendation_api(model_name: str, crypto: str, last_n: int = 500):
    history  = get_recommendation_history(model_name, crypto, last_n)
    if history  is not None:
        return history 
    else:
        return {"error": "No recommendations found"}
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)