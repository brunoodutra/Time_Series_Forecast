# Trade Bot

Bot de trading para Binance com leitura de sinais via API local, suporte a `PAPER` e `REAL`, e operacao em `FUTURES` ou `SPOT`.

## Visao Geral

O bot executa um loop continuo para:

1. Sincronizar o estado local das operacoes.
2. Buscar a ultima recomendacao da API de sinais.
3. Decidir se deve abrir, manter ou fechar posicao.
4. Persistir estado e historico estruturado de eventos.

Os sinais sao lidos da API em `RECOMMENDATION_API_URL`, normalmente exposta pelo projeto em `finance/API/API_setup.py`.

## Arquitetura

- `src/main.py`
  - ponto de entrada do bot
  - inicializa exchange, estado, leitura de sinais e loop principal

- `src/config.py`
  - carrega configuracoes do `.env`
  - define modos de operacao, simbolos e parametros de risco

- `src/exchange_client.py`
  - encapsula a integracao com a Binance
  - suporta:
    - `MARKET_MODE=FUTURES` com `ccxt.binanceusdm`
    - `MARKET_MODE=SPOT` com `ccxt.binance`
  - normaliza simbolos automaticamente

- `src/signal_reader.py`
  - consome a API de recomendacoes
  - busca `last_recommendation`
  - busca `last_target_stop` quando o sinal nao e `Hold`

- `src/trade_manager.py`
  - regra principal de decisao
  - abre posicao, monitora entrada, cria protecoes e fecha operacoes

- `src/state_manager.py`
  - persiste o estado atual dos trades
  - persiste tambem um historico estruturado de eventos para monitoramento

- `run_bot.py`
  - launcher simples do bot

## Modos de Operacao

### Trading Mode

- `TRADING_MODE=PAPER`
  - simula ordens localmente
  - nao envia ordens reais
  - util para validar fluxo

- `TRADING_MODE=REAL`
  - envia ordens reais para a Binance
  - exige credenciais validas

### Market Mode

- `MARKET_MODE=FUTURES`
  - usa Binance USD-M Futures
  - permite `buy` e `sell` como abertura de posicao
  - usa ordens de protecao separadas de stop e take profit

- `MARKET_MODE=SPOT`
  - usa carteira spot da Binance
  - `Buy` abre posicao comprada
  - `Sell` fecha posicao comprada existente
  - nao abre short
  - tenta criar protecao via OCO quando a compra e confirmada

## Persistencia e Monitoramento

O bot usa dois arquivos em `data/`:

- `active_trades.json`
  - estado atual das operacoes conhecidas
  - guarda trades `PENDING`, `OPEN`, `PARTIALLY_FILLED` e `CLOSED`

- `order_history.json`
  - historico estruturado de eventos para auditoria e monitoramento
  - cada evento inclui `timestamp`, `event_type`, `symbol`, `status`, `details` e um snapshot resumido do trade

### Exemplos de eventos registrados

- `ENTRY_SUBMITTED`
- `ENTRY_ACCEPTED`
- `ENTRY_FILLED`
- `ENTRY_FAILED`
- `FUTURES_PROTECTION_CREATED`
- `SPOT_PROTECTION_CREATED`
- `STOP_LOSS_TRIGGERED`
- `TAKE_PROFIT_TRIGGERED`
- `REVERSAL_SIGNAL`
- `TRADE_CLOSED`
- `TRADE_SKIPPED`
- `STALE_PAPER_STATE_CLOSED`

Esse historico complementa os logs textuais da pasta `logs/`.

## Configuracao

Crie um arquivo `.env` na raiz de `finance/Trade_Bot`.

Exemplo para spot real:

```env
BINANCE_API_KEY=sua_api_key
BINANCE_SECRET_KEY=sua_secret_key
TRADING_MODE=REAL
MARKET_MODE=SPOT
LOG_LEVEL=INFO
RECOMMENDATION_API_URL=http://127.0.0.1:8000
RECOMMENDATION_MODEL_NAME=CNN
API_TIMEOUT_SECONDS=10
```

Exemplo para futures real:

```env
BINANCE_API_KEY=sua_api_key
BINANCE_SECRET_KEY=sua_secret_key
TRADING_MODE=REAL
MARKET_MODE=FUTURES
LOG_LEVEL=INFO
RECOMMENDATION_API_URL=http://127.0.0.1:8000
RECOMMENDATION_MODEL_NAME=CNN
API_TIMEOUT_SECONDS=10
```

Exemplo para simulacao:

```env
TRADING_MODE=PAPER
MARKET_MODE=SPOT
LOG_LEVEL=INFO
RECOMMENDATION_API_URL=http://127.0.0.1:8000
RECOMMENDATION_MODEL_NAME=CNN
API_TIMEOUT_SECONDS=10
PAPER_BALANCE=1000
```

## Parametros Principais

Em `src/config.py`:

- `SYMBOLS`
  - lista de ativos monitorados

- `CONFIDENCE_THRESHOLD`
  - confianca minima para aceitar o sinal

- `RISK_PER_TRADE`
  - percentual do saldo usado por operacao

- `LEVERAGE`
  - multiplicador de exposicao em futures
  - em spot, nao cria short, mas ainda influencia o calculo caso o codigo seja alterado para usar derivativos

## Fluxo do Bot

1. Valida configuracao.
2. Cria cliente da Binance.
3. Valida a conexao com o mercado configurado.
4. Carrega estado local.
5. Entra em loop:
   - sincroniza trade ativo por simbolo
   - consulta a API de recomendacao
   - executa a decisao
   - persiste estado e historico

## Regras de Decisao

### Futures

- `Buy` sem posicao -> abre long
- `Sell` sem posicao -> abre short
- sinal inverso com posicao aberta -> fecha a posicao atual

### Spot

- `Buy` sem posicao -> compra
- `Sell` com posicao comprada -> vende e fecha
- `Sell` sem posicao -> ignora

## Como Rodar

1. Suba a API local de recomendacoes.
2. Instale dependencias:

```bash
pip install -r requirements.txt
```

3. Execute o bot:

```bash
python run_bot.py
```

## Observacoes Importantes

- Em `REAL`, use credenciais da Binance compativeis com o mercado escolhido.
- Para `FUTURES`, a chave precisa de permissao de futures.
- Para `SPOT`, a chave precisa de permissao de trade spot.
- Nao habilite saques por API para esse bot.
- Se houver whitelist de IP, seu IP atual precisa estar autorizado.

## Limitacoes Atuais

- O historico e salvo em JSON, nao em banco de dados.
- OCO em spot depende das regras da Binance e do suporte do `ccxt`.
- Ainda nao existe painel visual de monitoramento; o acompanhamento e por `logs/`, `active_trades.json` e `order_history.json`.
