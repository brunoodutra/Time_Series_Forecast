# Exportação TFLite (quantizado)

Este documento explica como usar o script `export_tflite_quantized.py` para converter um modelo Keras/SavedModel em um arquivo TFLite, com suporte a diferentes modos de quantização.

## Objetivo
- Converter modelos treinados para um formato leve (`.tflite`) para uso em produção, dispositivos móveis ou ambientes com restrição de recursos.
- Suporte a quatro modos:
  - `float32`: sem quantização (arquivos maiores, máxima precisão)
  - `dynamic`: quantização de faixa dinâmica (boa relação tamanho/precisão, não requer dataset representativo)
  - `fp16`: pesos em `float16` (reduz tamanho com pequena perda)
  - `int8`: quantização inteira (máxima compactação, requer dataset representativo)

## Requisitos
- Python 3.8+
- TensorFlow 2.x
- Acesso à internet para o modo `int8` (coleta de dados do Binance para o dataset representativo)
- Diretório do modelo (`--model_dir`) acessível e compatível com `tf.keras.models.load_model`
- (Opcional, recomendado) `config.json` na pasta do modelo, contendo parâmetros do pipeline de features/normalização

## Local do script
- `finance/AI/Classification/Experiments/Cryptos/export_tflite_quantized.py`

## Onde encontrar o modelo
- Use a pasta onde seu modelo Keras foi salvo (SavedModel ou `.h5`). Exemplo típico:
  - `finance/AI/Classification/Real_Time_Inference/<modelo_BTC_4h>/` (ajuste ao seu caso)

## Uso básico
### Dynamic range quantization
```
python finance/AI/Classification/Experiments/Cryptos/export_tflite_quantized.py \
  --model_dir "finance/AI/Classification/Real_Time_Inference/<pasta_do_modelo>" \
  --output_dir "tflite_exports" \
  --quantization dynamic
```

### FP16
```
python finance/AI/Classification/Experiments/Cryptos/export_tflite_quantized.py \
  --model_dir "finance/AI/Classification/Real_Time_Inference/<pasta_do_modelo>" \
  --output_dir "tflite_exports" \
  --quantization fp16
```

### INT8 (com dataset representativo)
```
python finance/AI/Classification/Experiments/Cryptos/export_tflite_quantized.py \
  --model_dir "finance/AI/Classification/Real_Time_Inference/<pasta_do_modelo>" \
  --output_dir "tflite_exports" \
  --quantization int8 \
  --symbol BTC \
  --interval 4h \
  --samples 128
```

### Float32 (sem quantização)
```
python finance/AI/Classification/Experiments/Cryptos/export_tflite_quantized.py \
  --model_dir "finance/AI/Classification/Real_Time_Inference/<pasta_do_modelo>" \
  --output_dir "tflite_exports" \
  --quantization float32
```

## Parâmetros
- `--model_dir`: diretório do modelo salvo (SavedModel ou `.h5`).
- `--output_dir`: pasta para salvar o `.tflite` (criada se não existir).
- `--quantization`: `float32`, `dynamic`, `fp16` ou `int8`.
- `--symbol`: símbolo cripto para coletar dados do dataset representativo (`int8`).
- `--interval`: intervalo dos candles (ex.: `1h`, `4h`, `1d`).
- `--samples`: número de amostras para o dataset representativo (`int8`).

## Saída
- O arquivo é salvo em: `--output_dir/model_<quantization>.tflite`.

## Verificação rápida (carregar e inferir com TFLite)
### Float/dynamic/fp16
```python
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="tflite_exports/model_dynamic.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
x = np.random.rand(*input_shape).astype(np.float32)  # Exemplo

interpreter.set_tensor(input_details[0]['index'], x)
interpreter.invoke()
y = interpreter.get_tensor(output_details[0]['index'])
print(y.shape, y.dtype)
```

Resumo: exemplo mínimo para carregar o modelo TFLite e executar uma inferência com entradas float.

### INT8 (respeitando escala e zero-point)
```python
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="tflite_exports/model_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Obtém quantização do input
scale, zero_point = input_details[0]['quantization']
input_shape = input_details[0]['shape']

# Supondo que você já tenha x_float normalizado (ex.: min-max [0,1])
x_float = np.random.rand(*input_shape).astype(np.float32)
x_int8 = (x_float / scale + zero_point).astype(np.int8)

interpreter.set_tensor(input_details[0]['index'], x_int8)
interpreter.invoke()
y_int8 = interpreter.get_tensor(output_details[0]['index'])
print(y_int8.shape, y_int8.dtype)
```

Resumo: demonstra como converter entradas float normalizadas para int8 usando escala/zero-point do modelo e realizar uma inferência.

## Gerar `x_float` com seu pipeline
Para garantir consistência entre treino/inferência, você pode usar o mesmo pipeline:
```python
import numpy as np
import os, sys
sys.path.append(os.path.abspath("./finance/AI/Processing/"))

from DataLoaderPipeline import FeaturesDataGenerator, scrapingHistoricalData
from datetime import datetime, timedelta

# Parâmetros do seu config.json (exemplo)
parameters = {
    "datatype": "2D",
    "lookback": 64,
    "pred_days": 0,
    "features_indicators": ["EMA", "MACD", "RSI", "BBANDS"],
    "min_norm": 0.0,
    "max_norm": 1.0,
}

symbol = "BTC"
interval = "4h"

SHD = scrapingHistoricalData()
start_time = (datetime.today() - timedelta(days=60)).strftime('%Y-%m-%d')
data_df = SHD.get_crypto_historical_data([symbol], interval, start_time)

dataGen = FeaturesDataGenerator(
    data_df,
    datatype=parameters['datatype'],
    lookback=parameters['lookback'],
    pred_days=parameters['pred_days'],
    shuffle=False,
    batch_size=32,
    selected_features=parameters['features_indicators'],
    data_augmentation=False,
    min_max_norm_features=[parameters['min_norm'], parameters['max_norm']],
)

x_inf = dataGen.comput_features(data_df, pred_days=0)
x_float = dataGen.apply_NomrMinmax(x_inf, parameters['min_norm'], parameters['max_norm'], axis=0)

if parameters['datatype'] == '2D':
    x_float = np.transpose(x_float, [0, 2, 1]).reshape(-1, dataGen.inputShape[1], dataGen.inputShape[2], 1)

print(x_float.shape, x_float.dtype)  # Use amostras de x_float nas inferências
```

Resumo: produz `x_float` com o mesmo pipeline de features/normalização usado no treino para garantir compatibilidade com o modelo.

## Observações importantes
- `int8` requer dataset representativo; o script coleta ~60 dias de dados para calibrar.
- Mantenha o mesmo pré-processamento (features/normalização) entre treino e inferência.
- `tf.keras.models.load_model` precisa de objetos customizados do projeto (métricas/perdas), já tratados pelo script.
- Para `int8`, verifique `input_details[0]['quantization']` para converter inputs corretamente.
- O formato/shape de entrada depende do seu modelo (`datatype` 1D/2D e `lookback`/`features`).

## Erros comuns e soluções
- `config.json` ausente: use quantização `dynamic/fp16/float32` ou forneça parâmetros manualmente.
- Falha ao baixar dados: verifique internet e disponibilidade do endpoint da Binance.
- Incompatibilidade de input shape: garanta que `x_float` tenha o shape esperado pelo modelo.
- Operações não suportadas no TFLite: revise arquitetura (camadas/ops) ou mude o modo de quantização.

## Integração com produção
- O arquivo `.tflite` pode ser carregado por `tf.lite.Interpreter` em sua aplicação de produção.
- Para manter consistência, centralize sua engenharia de features usando `FeaturesDataGenerator` também no ambiente de produção.