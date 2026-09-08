# 📈 Analisi Tecnica Borsa AI

Prima versione sperimentale di un'app personale di analisi tecnica.

## Obiettivo

L'app analizza automaticamente i titoli azionari e produce:

- Trend Score 1-10
- Reversal Risk 0-10
- analisi multi-timeframe
- Heikin-Ashi
- Parabolic SAR
- Bollinger Bands
- MACD
- RSI
- Stochastic RSI
- Chaikin Oscillator
- Advance/Decline
- Volume Oscillator
- ADX
- divergenze
- Bollinger Detachment
- Middle Bollinger Break
- segnali anticipatori rispetto al SAR
- Price Target medio
- Rendimento Atteso
- Beta

## Mercati iniziali

### Italia

- ENI
- Intesa Sanpaolo
- Saipem
- Enel
- Leonardo
- Generali
- STMicroelectronics
- Unicredit

### Europa

- SAP
- Siemens
- LVMH
- Airbus
- ASML
- Allianz
- TotalEnergies

### USA

- Apple
- Microsoft
- Nvidia
- Amazon
- Alphabet
- Meta
- Tesla
- JPMorgan

## Timeframe

L'app utilizza:

- Weekly
- Daily
- 4H
- 2H
- 1H

Il timeframe Daily è quello principale.

Weekly e Daily sono analizzati tramite dati giornalieri.

4H e 2H vengono ricostruiti a partire dai dati orari.

## Pesi

### Parabolic SAR

- Weekly: 15%
- Daily: 15%
- 4H: 5%

Totale SAR = 35%

### Altri indicatori

- Bollinger + Heikin-Ashi: 10%
- MACD: 10%
- RSI: 5%
- Stochastic RSI: 5%
- Chaikin: 10%
- Advance/Decline: 10%
- Volume Oscillator: 10%
- ADX: 5%

Totale = 100%

## Regola fondamentale

Il SAR può arrivare in ritardo.

Per questo motivo l'app cerca segnali anticipatori attraverso:

- Bollinger Detachment
- Heikin-Ashi
- MACD
- RSI
- Stochastic RSI
- Chaikin
- Advance/Decline
- Volume Oscillator
- divergenze

Esempio:

Bollinger superiore
+
distacco delle candele Heikin-Ashi
+
candela di indecisione
+
MACD in deterioramento
+
Chaikin in calo

può generare un PRE-ALERT anche se il SAR Daily è ancora rialzista.

Se successivamente il SAR conferma, il livello di attenzione aumenta.

## Bollinger

La banda superiore e inferiore vengono utilizzate per cercare il "distacco".

Distacco dalla banda superiore:

possibile anticipazione di indebolimento/inversione ribassista.

Distacco dalla banda inferiore:

possibile anticipazione di recupero/inversione rialzista.

La Middle Band viene invece trattata come possibile supporto/resistenza dinamica.

La sua rottura viene considerata una conferma più forte del movimento.

## Heikin-Ashi

Le candele Heikin-Ashi vengono utilizzate per identificare:

- direzione
- cambio di colore
- indecisione
- perdita di momentum

Una candela con corpo molto piccolo rispetto al range viene classificata come candela di indecisione.

## MACD

Parametri:

12 / 26 / 9

Particolare importanza:

Bullish crossover sotto lo zero:

segnale rialzista forte.

Bearish crossover in area elevata:

segnale di indebolimento.

## RSI

Periodo:

14

Non viene utilizzata una semplice regola:

RSI > 70 = vendita

RSI < 30 = acquisto

Il sistema considera anche direzione e variazione.

## Stochastic RSI

Parametri iniziali:

14 / 14 / 3 / 3

Serve principalmente per il momentum di breve periodo.

## Chaikin

Il sistema considera:

- valore
- direzione
- eccesso
- deterioramento
- divergenze

Un Chaikin molto elevato che inizia a diminuire può rappresentare un segnale di possibile esaurimento del movimento rialzista.

## Advance/Decline

La prima versione utilizza la variazione giornaliera dei titoli dell'universo del mercato selezionato.

Successivamente il modulo dovrà essere migliorato con un vero indicatore di market breadth.

## Volume Oscillator

Parametri:

14 / 28

Serve per verificare se i volumi confermano il movimento del prezzo.

## ADX

Periodo:

14

Serve principalmente a misurare la forza del trend e non la sua direzione.

## Price Target / Rendimento Atteso / Beta

Questi dati vengono mostrati come informazioni aggiuntive.

NON partecipano al Trend Score.

Rendimento Atteso:

(Price Target - Prezzo Attuale) / Prezzo Attuale × 100

## Portfolio

La sidebar contiene:

PORTFOLIO ON/OFF

ON significa che il titolo è considerato in posizione LONG.

La prima versione utilizza il controllo solamente come informazione.

Nelle versioni successive il sistema dovrà aumentare la priorità degli alert sui titoli Portfolio ON.

Non vengono memorizzati:

- capitale investito
- quantità
- prezzo di carico
- profit/loss

## Installazione

È consigliato utilizzare Python 3.11 o superiore.

Nel terminale:

```bash
pip install -r requirements.txt