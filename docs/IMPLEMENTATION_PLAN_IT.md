# Piano di Implementazione — Forecasting Probabilistico di Serie Temporali con Modelli di Diffusione
### Progetto d'esame di PML (Probabilistic Machine Learning, apprendimento automatico probabilistico) · Università di Trieste · Prof. Luca Bortolussi · Gruppo di 3

> **Stato:** è solo un documento di pianificazione. Nessun codice, per ora. È il progetto-guida che il gruppo eseguirà in tre passi.
> **Nota sulle pagine del PDF:** il PDF degli appunti ha uno scarto di +10 rispetto ai numeri di pagina stampati (pagina stampata X = pagina PDF X+10). Le citazioni qui sotto usano i **numeri di sezione** (non ambigui) con la pagina stampata fra parentesi.
> **Nota linguistica:** i termini tecnici standard (diffusion model, forecasting, dataset, deep learning, baseline, sampling, ecc.) restano in inglese, perché sono così nella letteratura e negli appunti del professore. Ogni sigla viene sciolta per esteso la prima volta che compare.

---

## Parte 0 — Come usare questo documento

**Che cos'è.** Un piano completo e pronto all'esecuzione: la domanda di ricerca, il legame preciso con il corso di PML, i modelli, i dataset, le metriche, gli esperimenti, l'architettura software, il flusso di lavoro del gruppo in 3 passi, i rischi e i prodotti finali (slide + preparazione orale). È scritto in modo che ogni membro del gruppo possa aprirlo e capire subito cosa fare.

**Che cosa NON è.** Non è la relazione, non è il codice, non sono le slide. Quelli arrivano dopo, *a partire da* questo piano.

**Come leggerlo.**
- Le Parti 1–3 si leggono insieme, come gruppo, all'inizio. Bisogna accordarsi sulla *storia* da raccontare.
- Poi ciascuno cura le parti assegnate al proprio ruolo (Parte 9).
- La Parte 7 (esperimenti) è la fonte di verità su "che cosa significa aver finito".
- L'Appendice A è una proposta già pronta da inviare al professore.
- L'Appendice E spiega in profondità la fase sperimentale (prima la metodologia, poi le istruzioni operative passo-passo).

**La regola d'oro del progetto:** *non stiamo presentando una rassegna bibliografica e non stiamo costruendo una scatola nera.* Prendiamo una domanda probabilistica ben definita, le rispondiamo con esperimenti veri, e siamo capaci di aprire ogni modello che usiamo e spiegarlo con la notazione stessa del professore.

---

## Parte 1 — Il progetto in una pagina

### 1.1 Domanda di ricerca (una frase)
> Un **diffusion model** può produrre **forecast probabilistici** di una serie temporale futura — una *distribuzione* di futuri plausibili invece di un singolo numero — che siano più **accurati**, meglio **calibrati** o più **informativi** rispetto alle baseline classiche e di deep learning, e a quale **costo computazionale**?

### 1.2 Abstract (la versione in 6 frasi)
Il forecasting reale non è "la domanda di energia elettrica di domani sarà 100"; è "ecco la distribuzione dei domani plausibili". Inquadriamo il forecasting multi-step come l'apprendimento della **distribuzione generativa condizionata** `p(futuro | passato)`, e la realizziamo con un **diffusion model condizionato**, addestrato con lo stesso obiettivo — ELBO (Evidence Lower BOund, limite inferiore dell'evidenza) / predizione del rumore — che il corso ricava nella §11.2. Lo confrontiamo testa a testa con una baseline ingenua, un modello statistico classico (ARIMA/ETS) e un modello probabilistico autoregressivo di deep learning (DeepAR), su un dataset reale multivariato. Valutiamo le **tre** cose che l'esame premia: **accuratezza puntuale** (MAE/RMSE), **qualità probabilistica** (CRPS, copertura, calibrazione) e **costo** (tempo di addestramento/inferenza, e il compromesso qualità-vs-passi di denoising). La nostra tesi è volutamente non trionfalistica: la diffusione regala un'incertezza *più ricca e meglio calibrata*, ma il suo vantaggio dipende dall'orizzonte, dal dataset e da un costo reale di sampling. Il prodotto finale è una presentazione di 8 minuti e un repository pulito che dimostra che abbiamo capito il corso *e* ne abbiamo esteso l'ultimo capitolo in un'applicazione funzionante.

### 1.3 La tesi che difenderemo (la "storia")
**"Per il forecasting, la domanda giusta non è *cosa* succederà ma *cosa potrebbe* succedere e *quanto ne siamo sicuri*. I diffusion model rispondono nativamente a questa seconda domanda — generano campioni del futuro — ma il premio è la qualità dell'incertezza, non un MAE più basso, e questo premio ha un costo di sampling che possiamo misurare e regolare."**

Questa storia è forte perché è *falsificabile* (forse la diffusione perde — è comunque un risultato), *probabilistica* (vive nel cuore della PML) e *pratica* (il compromesso sul costo è una preoccupazione reale di chi mette in produzione un modello).

### 1.4 Criteri di successo, mappati sulla griglia di valutazione del professore
| Criterio di valutazione (dalla descrizione d'esame) | Come il progetto ci si posiziona |
|---|---|
| Chiarezza e completezza dell'esposizione | 8 slide essenziali, un solo messaggio ciascuna (Parte 11); un repository pulito; risultati onesti. |
| **Originalità rispetto ai contenuti del corso** | Estendiamo la §11.2 (diffusione) dalla *generazione non condizionata* al *forecasting condizionato* — esattamente l'impostazione della survey. In più, un'analisi incertezza aleatoria-vs-epistemica (§1.4 qui) che il corso imposta ma non applica alla diffusione. |
| Comprensione di teoria e pratica | L'intera pipeline è mappata sui capitoli del corso (Parte 2); sappiamo ricavare la loss della diffusione nella notazione del professore; ne abbiamo costruito (almeno una versione giocattolo) noi stessi. |
| Chiarezza e precisione all'orale | Preparazione orale per persona, con le domande probabili mappate sulle pagine del corso (Parte 11.3). |

### 1.5 Decisioni di perimetro (fissate, dalle domande di pianificazione)
- **Tempistiche:** flessibili → il piano è **modulare**, con livelli "fermati-dopo" (Parte 9.4). Ci si può fermare dopo il Livello 0 e avere comunque un progetto completo.
- **Calcolo:** Colab Pro / GPU (Graphics Processing Unit, unità di elaborazione grafica) a pagamento → modelli del livello di TimeGrad sono fattibili; lo mettiamo a budget (Parte 8.2).
- **Perimetro:** **bilanciato** → un diffusion model di frontiera + baseline complete + valutazione probabilistica + **un esperimento di robustezza** (cambio di regime). Un secondo dataset / un secondo diffusion model sono *facoltativi*, non obbligatori.
- **Competenze del gruppo:** a proprio agio con Python, nuovi ai framework di deep learning → **prima le librerie** (GluonTS / PyTorchTS / Darts / statsforecast), con un piccolo **DDPM giocattolo scritto da zero** come "artefatto di comprensione" (Parte 5.4). Usiamo le librerie per ottenere risultati e ne leggiamo le interiora per non usarle come scatola nera.

---

## Parte 2 — Legame con il corso di PML (la spina dorsale teorica)

Questo è l'elemento di differenziazione di maggior valore per il voto. Il corso *finisce* sulla diffusione (Cap. 11): quindi stiamo estendendo l'ultimo capitolo, non importando qualcosa di estraneo. Ogni componente del progetto si mappa su un capitolo.

### 2.1 La tabella di mappatura (componente del progetto → corso → cosa dire all'orale)
| Componente del progetto | Posizione nel corso (pag. stampata) | Risposta orale in una riga |
|---|---|---|
| Forecasting come `p(futuro\|passato)` | §1.2 Generative Modelling (p.3–4) | "È modellazione generativa condizionata: impariamo una distribuzione, poi campioniamo futuri." |
| Perché una *distribuzione* e non un punto | §1.1.1 Incertezza, aleatoria vs epistemica (p.2–3) | "Le decisioni hanno bisogno dell'incertezza; una stima puntuale nasconde il rischio." |
| Processo forward di diffusione | §11.2.1 (p.123–124) | "Una catena di Markov gaussiana fissa che corrompe i dati in rumore bianco: `p(x_t\|x_{t-1})=N(√(1−β_t)x_{t-1},β_tI)`." |
| Rumore in forma chiusa | §11.2.1 (p.124) | "`x_t=√ᾱ_t x_0+√(1−ᾱ_t)ε` — campiona qualunque passo in un colpo solo, col reparameterization trick." |
| Processo reverse / generativo | §11.2.2 (p.124–125) | "Si imparano i kernel gaussiani inversi `q_θ(x_{t-1}\|x_t)`; valido perché un rumore lento ⇒ inverso ~gaussiano." |
| Obiettivo di addestramento | §11.2.2–11.2.3 (p.125–126) | "Minimizzare `KL(forward‖backward)` = massima verosimiglianza; in pratica l'MSE di predizione del rumore." |
| La loss che ottimizziamo davvero | §11.2.3 (p.126), Alg. 1 | "`E‖ε − ε_θ(√ᾱ_t x_0+√(1−ᾱ_t)ε, t)‖²`." |
| Campionare un forecast | §11.2.3 Alg. 2 (p.126); ancestral sampling §3.3.1 (p.26) | "Si parte dal rumore e si percorre la catena di Markov inversa; ogni esecuzione = una traiettoria futura." |
| Macchinario ELBO (condiviso col VAE) | §9.2 ELBO (p.96); §11.1 VAE (p.119–122) | "Lo stesso limite inferiore del VAE: ricostruzione − KL verso il prior." |
| Reparameterization trick | §10.4.1 (p.112–113); usato in §11.1 (p.122) | "Si sposta la casualità su `ε~N(0,I)` perché i gradienti attraversino il sampling." |
| Vista score-based / Langevin | §11.2.5 (p.127–128); sampling Cap. 8 (p.79–91) | "La diffusione ≈ stimare `∇_x log p(x)` e fare sampling tipo Langevin — si lega a MCMC." |
| Baseline DeepAR (verosimiglianza + sampling) | §1.2 (p.3–4); distrib. predittiva §6.5 (p.64) | "Modello a verosimiglianza autoregressiva; le simulazioni Monte-Carlo danno la distribuzione predittiva." |
| Baseline classica ARIMA/ETS | spazio degli stati ≈ HMM Cap. 5 (p.49); regr. lin. bayesiana Cap. 6 | "Dinamica lineare-gaussiana; intervalli predittivi dal modello di rumore gaussiano." |
| CRPS / scoring proprio | KL & scoring §2.3 (p.20) | "Il CRPS generalizza il MAE alle distribuzioni; è una regola di scoring propria." |
| (Facoltativo) baseline GP | Cap. 12 Kernels & GP (p.129+, con asterisco) | "Un forecaster pienamente bayesiano con incertezza epistemica — il caso di contrasto." |

> **Sigle sciolte in questa tabella:** MSE (Mean Squared Error, errore quadratico medio); VAE (Variational AutoEncoder, autoencoder variazionale); KL (Kullback–Leibler, la divergenza); MCMC (Markov Chain Monte Carlo, Monte Carlo a catena di Markov); HMM (Hidden Markov Model, modello di Markov nascosto); ARIMA (AutoRegressive Integrated Moving Average); ETS (Error, Trend, Seasonality — lisciamento esponenziale); GP (Gaussian Process, processo gaussiano).

### 2.2 L'unica derivazione che dobbiamo padroneggiare (nella notazione del professore)
La loss di addestramento della diffusione è un ELBO ri-parametrizzato. La catena che il gruppo deve saper riprodurre alla lavagna:

1. **Forward (fisso):** `p(x_t|x_{t-1}) = N(x_t; √(1−β_t)·x_{t-1}, β_t·I)`, con `α_t = 1−β_t`, `ᾱ_t = ∏_{i≤t} α_i`.
2. **Forma chiusa (reparameterization trick, §10.4.1):** `x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε`, `ε ~ N(0,I)` ⟹ `p(x_t|x_0)=N(√ᾱ_t x_0, 1−ᾱ_t)`; per `t→∞`, `→ N(0,I)`.
3. **Reverse (appreso):** `q_θ(x_{t-1}|x_t) = N(μ_θ(x_t,t), Σ_θ(x_t,t))`.
4. **Obiettivo:** minimizzare `KL(p(x_{0:T}) ‖ q_θ(x_{0:T}))` `= −E[Σ_t log q_θ(x_{t-1}|x_t)] + cost` (= massima verosimiglianza / ELBO negativo).
5. **Reparametrizzazione per predizione del rumore (§11.2.3):** predire `ε` invece di `x_0` dà la loss semplice
   `L(θ) = E_{t,x_0,ε} ‖ ε − ε_θ(√ᾱ_t·x_0 + √(1−ᾱ_t)·ε, t) ‖²` (Algoritmo 1).
6. **Sampling (Algoritmo 2):** `x_{t-1} = (1/√α_t)(x_t − (1−α_t)/√(1−ᾱ_t)·ε_θ(x_t,t)) + σ_t·z`, `z~N(0,I)`.

**La nostra novità in una riga:** rendere il denoiser **condizionato sul passato**, `ε_θ(x_t, t, c)`, dove `c = Encoder(x_{passato})`. La survey lo chiama *fonte di condizionamento = serie storica* e *integrazione della condizione = come `c` entra nella rete*. Tutto il resto è la §11.2 del corso.

### 2.3 Il punto "originale + sofisticato" (scomposizione dell'incertezza)
Il corso distingue l'incertezza **aleatoria** (irriducibile, rumore dei dati) da quella **epistemica** (modello/parametri) (§1.1.1) e mostra come i modelli bayesiani (Cap. 6), le BNN (Bayesian Neural Network, reti neurali bayesiane, §10.6) e i GP (Cap. 12) catturino l'incertezza *epistemica* mettendo distribuzioni sui parametri. Un forecaster a diffusione con pesi `θ` stimati puntualmente cattura l'incertezza **aleatoria** (la dispersione dei futuri plausibili) ma **non** quella **epistemica** (è sicuro dei propri parametri). Questo ci dà una slide di discussione genuinamente originale — *"quale incertezza stiamo davvero quantificando, e cosa ci sfugge?"* — e un naturale aggancio per i "lavori futuri" (diffusione a ensemble / bayesiana). Pochi progetti d'esame fanno questa distinzione in modo esplicito; colpisce direttamente il criterio "profondità di comprensione".

---

## Parte 3 — Formalizzazione del problema

### 3.1 Notazione
- Una serie multivariata `x_{1:L} ∈ R^{D×L}` (`D` canali, lunghezza `L`).
- **Finestra di contesto / storia** `H`: il passato su cui condizioniamo, `c = x_{t-H+1 : t}`.
- **Orizzonte di previsione** `τ`: il futuro che prevediamo, `y = x_{t+1 : t+τ}`.
- Un **forecast** è la distribuzione condizionata `p(y | c)`. Le diverse classi di modelli la rappresentano in modi diversi (Parte 5).

### 3.2 Il compito, con precisione
Stimare e campionare da `p(x_{t+1:t+τ} | x_{t-H+1:t})`. Per la diffusione e DeepAR estraiamo `S` traiettorie campione `{y^{(s)}}_{s=1..S}`; per ARIMA/ETS otteniamo intervalli predittivi parametrici; per la baseline ingenua otteniamo un punto (e una banda empirica via bootstrap dei residui).

### 3.3 Forecasting deterministico vs probabilistico (l'inquadramento che ribadiamo sempre)
- **Deterministico:** restituisce `ŷ` (un numero). Valutato con MAE/RMSE.
- **Probabilistico:** restituisce `p(y|c)` (una distribuzione). Valutato con CRPS, copertura, calibrazione — *e* da esso puoi comunque estrarre un punto (la media/mediana) per MAE/RMSE, così il confronto è equo.

### 3.4 Rappresentazione del forecast per ciascun modello (perché il confronto sia "mele con mele")
| Modello | Output nativo | Forecast puntuale = | Intervallo predittivo = |
|---|---|---|---|
| Seasonal-naive | punto | il valore stesso | banda da bootstrap dei residui |
| ARIMA/ETS | predittiva gaussiana | media | intervallo analitico ±z·σ |
| DeepAR | traiettorie campionate | media/mediana dei campioni | quantili dei campioni |
| Diffusione (TimeGrad) | traiettorie campionate | media/mediana dei campioni | quantili dei campioni |

> **Sigle:** MAE (Mean Absolute Error, errore assoluto medio); RMSE (Root Mean Squared Error, radice dell'errore quadratico medio); CRPS (Continuous Ranked Probability Score, punteggio di probabilità classificato continuo).

---

## Parte 4 — Dati

Manteniamo il *piano* agnostico rispetto al dataset (come richiesto) ma vincoliamo la *scelta* perché il progetto resti rigoroso e i dati restino interessanti e riutilizzabili, non da manuale/noiosi.

### 4.1 Griglia di selezione del dataset (scegline uno che soddisfi tutti questi punti)
1. **Multivariato** (`D ≥ ~8`) così che la modellazione congiunta della diffusione abbia senso.
2. **Ha numeri pubblicati di TimeGrad/CSDI** → abilita E0 (il cancello "riproduci-un-risultato-noto"). È il vincolo più importante per la credibilità.
3. **Campionamento regolare, dati reali, rilevanza industriale** → competenza riutilizzabile, bella storia.
4. **Entra nella memoria/tempo di Colab** con un sottoinsieme sensato (poche centinaia di serie al massimo, o un sottoinsieme di canali).
5. **Ha un plausibile "cambio di regime"** sfruttabile per l'esperimento di robustezza E4 (stagionalità, una rottura in epoca COVID, un cluster di volatilità, ecc.).

### 4.2 Rosa dei candidati (interessanti + riutilizzabili + già benchmark)
| Dataset | D / cadenza | Perché interessante / riutilizzabile | Nel benchmark TimeGrad? |
|---|---|---|---|
| **Solar** (produzione fotovoltaica, 137 stazioni) | 137 / oraria | energia rinnovabile, intermittenza, forte ciclo giornaliero | ✅ |
| **Electricity** (370 clienti) | 370 / oraria | carico industriale, classico, facile da raccontare | ✅ |
| **Traffic** (San Francisco, 963 sensori) | 963 / oraria | spazio-temporale, occupazione stradale | ✅ |
| **Exchange** (8 valute) | 8 / giornaliera | piccolo e veloce → ottimo per iterare / ripiego | ✅ |
| **Wind power** (SCADA / generazione) | varia | rinnovabile, guidato dal meteo, sapore "Industria 4.0" | ⚠️ varia |
| **ETT** (temperatura trasformatori elettrici) | 7 / 15-min·oraria | usato in tutti i paper moderni a orizzonte lungo | ⚠️ (forecasting, non sempre CRPS) |

> **Sigle:** PV/fotovoltaico (PhotoVoltaic); SCADA (Supervisory Control And Data Acquisition, controllo di supervisione e acquisizione dati); ETT (Electricity Transformer Temperature, temperatura del trasformatore elettrico); CSDI (Conditional Score-based Diffusion model for Imputation, modello di diffusione condizionato basato sullo score per l'imputazione).

### 4.3 Raccomandazione
- **Primario:** **Solar** *oppure* **Electricity** (multivariato, energia, nel benchmark TimeGrad → E0 possibile).
- **Ripiego veloce / per iterare:** **Exchange** (minuscolo; fai girare tutta la pipeline da capo a fondo qui prima, *poi* sali di scala).
- **"Spunto interessante" facoltativo (E5):** **Wind power** o **Traffic**.
- **Punto di decisione per il gruppo:** confermare un primario + Exchange-per-iterare. *Non* partire dal dataset grande; prima fai diventare verde la pipeline su Exchange.

### 4.4 Specifica di preprocessing (scrivila una volta in `src/data/`, riusala ovunque)
- **Split temporale, niente leakage:** `train | val | test` contigui in ordine di tempo (es. 70/10/20). Mai mescolare nel tempo. Il test è la fetta *più recente*.
- **Scaling:** addestra uno scaler (standardizzazione per-serie o mean-scaling alla GluonTS) **solo sul train**; applicalo a val/test. Il leakage di scaling è il bug silenzioso n.1 — proteggiti.
- **Finestratura:** finestre scorrevoli `(contesto H, orizzonte τ)` *dentro* ciascuno split; mai far attraversare a una finestra il confine dello split.
- **Valori mancanti / zeri:** documenta la politica (forward-fill per buchi corti, mascheramento per quelli lunghi); annota il significato fisico (un contatore a zero può voler dire "spento", non "domanda 0").
- **Frequenza & feature di calendario:** tienile minime all'inizio (ora-del-giorno, giorno-della-settimana) — contano per le covariate di DeepAR/TimeGrad.
- **Determinismo:** una sola funzione costruisce i dataset da un seed fisso; produce un `manifest.json` che registra le date di split, le statistiche dello scaler, `H`, `τ`.

### 4.5 Il "contratto sui dati"
Un unico loader restituisce dataset `(train, val, test)` in stile GluonTS + un dizionario di metadati (`D, freq, H, τ, scaler`). Ogni modello consuma lo *stesso* oggetto. Questo garantisce per costruzione che il confronto sia equo.

---

## Parte 5 — Modelli (la scala del confronto)

Cinque gradini, sofisticazione crescente. M0–M3 sono la scala **obbligatoria**; M4 e il GP sono **facoltativi**. Per ogni modello: il suo ruolo, la libreria, cosa configurare, cosa capire per l'orale, e il compito "apri-la-scatola" che gli impedisce di essere una scatola nera.

### 5.1 M0 — Naive / Seasonal-naive (l'ancora di onestà)
- **Ruolo:** la baseline "stupida ma essenziale". Se la diffusione non batte *questa*, è già di per sé un risultato da titolo.
- **Libreria:** Darts (`NaiveSeasonal`) o 5 righe di NumPy.
- **Versione probabilistica:** bootstrap dei residui → banda predittiva empirica (così ha anche un CRPS/copertura).
- **Punto orale:** "una seasonal-naive forte è una baseline notoriamente difficile nel forecasting; batterla va guadagnato."

### 5.2 M1 — Statistica classica (ARIMA/SARIMA o ETS)
- **Ruolo:** il gradino della tradizione statistica; mostra che conosciamo il forecasting pre-deep learning.
- **Libreria:** `statsforecast` (Nixtla) `AutoARIMA`/`AutoETS` — veloce, vettorizzato, adatto ai principianti; oppure Darts.
- **Configura:** periodo stagionale, ricerca automatica dell'ordine; produci intervalli predittivi analitici.
- **Punto orale:** ARIMA = spazio degli stati lineare-gaussiano ⇒ si collega all'HMM (Cap. 5) e alla distribuzione predittiva gaussiana (Cap. 6).
- **Apri-la-scatola:** ispeziona l'autocorrelazione dei residui; mostra dove l'assunzione gaussiana/lineare si rompe.

### 5.3 M2 — Probabilistico di deep learning (DeepAR)
- **Ruolo:** la baseline *equa* di deep learning — restituisce già una distribuzione, così la diffusione non è confrontata con una rete deterministica.
- **Libreria:** **GluonTS** (`DeepAREstimator`, backend PyTorch). Ben documentata, gestisce caricamento/addestramento/valutazione.
- **Configura:** verosimiglianza (Student-t/gaussiana), lunghezza del contesto, dimensione della RNN, epoche; `num_parallel_samples` per il sampling predittivo.
- **Punto orale:** la fattorizzazione autoregressiva della verosimiglianza + il roll-out Monte-Carlo = distribuzione predittiva; stesso spirito generativo della diffusione, meccanismo diverso.
- **Apri-la-scatola:** disegna le traiettorie campionate; mostra come la dispersione dei campioni = incertezza predittiva.

> **Sigle:** DeepAR (Deep AutoRegressive, autoregressivo profondo); RNN (Recurrent Neural Network, rete neurale ricorrente).

### 5.4 M3 — Frontiera: diffusione condizionata (TimeGrad) ← il pezzo forte
- **Ruolo:** il modello di cui parla tutto il progetto.
- **Libreria:** **PyTorchTS** (`TimeGradEstimator`, dell'autore stesso di TimeGrad, costruita su GluonTS). È l'implementazione canonica e riproducibile.
- **Cos'è:** diffusione autoregressiva — una RNN codifica la storia in uno stato nascosto `h_t` (il condizionamento `c`), e a ogni passo di previsione un **DDPM condizionato** denoisa un campione del prossimo vettore multivariato dato `h_t`. Esattamente la §11.2 resa condizionata.
- **Configura:** passi di diffusione `T` (es. 100), schedule `β`, dimensioni RNN/denoiser, epoche, `num_parallel_samples`.
- **Punto orale:** saper indicare la loss di denoising e dire "questa è la §11.2.3, condizionata su `h_t`".
- **Apri-la-scatola (obbligatorio — è così che evitiamo la 'scatola nera'):**
  - varia `T` (i passi di denoising) → E3;
  - cambia la schedule `β` (lineare vs coseno) → ablazione;
  - disegna gli stati intermedi di denoising di un forecast (rumore → traiettoria);
  - leggi il forward/loss di `TimeGradEstimator` nel codice e annotalo a fronte degli appunti.

### 5.5 L'"artefatto di comprensione": un DDPM giocattolo da zero (piccolo, alto valore)
Poiché il gruppo è nuovo ai framework di deep learning e l'orale premia la comprensione reale, costruite **un** DDPM condizionato minimale *da zero* su una versione *univariata* o a 1 canale, seguendo alla lettera gli **Algoritmi 1 & 2** degli appunti:
- denoiser = piccola MLP (Multi-Layer Perceptron, percettrone multistrato) o CNN (Convolutional Neural Network, rete neurale convoluzionale) 1D `ε_θ(x_t, t, c)`;
- `T ≈ 100`, `β` lineare;
- addestra con la loss esatta della §11.2.3; campiona con l'Alg. 2.
- ~150 righe, gira su CPU/Colab in pochi minuti.
**Perché:** è la prova più pulita possibile di comprensione, alimenta la storia passi-vs-qualità (E3) senza attriti di libreria, ed è "originale rispetto ai contenuti del corso" perché il corso non costruisce mai la versione *condizionata*. Segnalalo come facoltativo solo se il tempo è davvero agli sgoccioli.

### 5.6 (Facoltativo) M4 — CSDI, o un contrasto con GP
- **CSDI** (diffusione condizionata basata sullo score, NeurIPS'21): il forecasting-come-imputazione con self-attention. Più pesante; è il *secondo diffusion model* se andiamo al Livello 2.
- **Regressione GP** (Cap. 12): un forecaster pienamente bayesiano con incertezza *epistemica* — la spalla perfetta per la discussione sulla scomposizione dell'incertezza (§2.3). Economico su dati grandi quanto Exchange via `GPyTorch`/`scikit-learn`.

---

## Parte 6 — Protocollo di valutazione

La valutazione *è* il rigore del progetto. Misuriamo **tre** famiglie (puntuale · probabilistica · costo) + imponiamo igiene statistica.

### 6.1 Accuratezza puntuale
- **MAE**, **RMSE** (sulla **media/mediana** predittiva) e **MASE** (Mean Absolute Scaled Error, errore assoluto scalato medio — adimensionale, permette di confrontare onestamente fra serie/dataset). Facoltativo sMAPE (symmetric Mean Absolute Percentage Error, errore percentuale assoluto medio simmetrico).

### 6.2 Qualità probabilistica (il cuore di PML)
- **CRPS** — la metrica di punta; generalizza il MAE alle distribuzioni complete; è una regola di scoring propria (cita Gneiting & Raftery 2007). Per il caso multivariato riporta il **CRPS-sum** (somma i canali, poi calcola il CRPS) come fa TimeGrad, per comparabilità.
- **Copertura @ {50, 80, 90}%** — i valori reali cadono dentro gli intervalli previsti al tasso dichiarato?
- **Ampiezza dell'intervallo** — quanto è "affilato"; stretto *e* calibrato è l'obiettivo.
- **Calibrazione:** istogramma PIT (Probability Integral Transform, trasformata integrale di probabilità) / reliability diagram — la distribuzione prevista è statisticamente coerente con la realtà?
- **Libreria:** il `Evaluator` di GluonTS dà MASE, sMAPE, **weighted-quantile-loss (un proxy del CRPS)** e copertura già pronti. Verifica il CRPS su un caso giocattolo contro un'implementazione fatta a mano (igiene E0).

### 6.3 Costo / efficienza (il vantaggio pratico)
- **Tempo di addestramento** a cronometro (per dataset, hardware fisso) e **n. di parametri**.
- **Latenza di inferenza:** tempo per produrre `S` traiettorie campione per una finestra.
- **GPU-ore e una stima approssimata in €** (Colab/cloud) — è un'intuizione pratica reale e valutabile.
- **La curva specifica della diffusione:** qualità (CRPS) vs **numero di passi di denoising `T`** (E3). È qui che la famosa lentezza di sampling della diffusione diventa un *risultato misurato*, legandosi direttamente al limite dichiarato dalla survey.

### 6.5 Igiene statistica (non saltarla — è originalità a basso costo)
- **≥3 seed casuali** per ogni modello appreso; riporta **media ± deviazione standard**, non singole esecuzioni.
- Split, scaling, `H`, `τ` e finestre di test **identici** fra tutti i modelli.
- **Cancello di riproduzione E0:** prima di fidarti di un qualunque numero, riproduci un CRPS-sum pubblicato per una coppia (modello, dataset) entro una tolleranza ragionevole. Se non ci riesci, la pipeline è sbagliata — sistemala prima di procedere.
- **Regole di equità:** le metriche puntuali si calcolano dalla media predittiva per *tutti* i modelli; mai confrontare una diffusione messa a punto con una baseline non messa a punto; mai scegliere l'orizzonte migliore a posteriori.

### 6.6 Trappole di valutazione da cui guardarsi attivamente
Leakage di look-ahead · scaling addestrato sul test · leakage da finestra a cavallo · confronto iniquo fra modelli distribuzionali e puntuali · cherry-picking dell'orizzonte/seed favorevole · riportare il CRPS senza dire se è CRPS-sum o medio. Mettile nella checklist di code-review (Parte 8.6).

---

## Parte 7 — Esperimenti (numerati, falsificabili)

Ogni esperimento dichiara un'**ipotesi**, un **setup**, **cosa variamo/misuriamo**, il **risultato che supporterebbe o smentirebbe la tesi** e l'**artefatto** (tabella/grafico) che produce. Questa è la definizione di "fatto". *(L'Appendice E spiega questa fase in profondità: prima la metodologia, poi il runbook operativo.)*

### E0 — Validazione della pipeline (cancello)
- **Ipotesi:** la nostra pipeline riproduce un risultato noto.
- **Setup:** un modello (DeepAR o TimeGrad) su un dataset già a benchmark (Electricity/Solar/Exchange).
- **Misura:** CRPS-sum vs il valore pubblicato.
- **Condizione di superamento:** entro tolleranza (diciamo stesso ordine di grandezza, ≲20%). **Finché E0 non passa, nessun altro numero è affidabile.**
- **Artefatto:** una singola riga di "riproduzione" + una nota sugli scarti residui.

### E1 — Confronto principale (il risultato centrale)
- **Ipotesi:** la diffusione è competitiva sull'accuratezza puntuale e *migliore sulla qualità probabilistica* rispetto alle baseline.
- **Setup:** M0,M1,M2,M3 sul dataset primario, `H`, `τ` fissi, ≥3 seed.
- **Misura:** MAE/RMSE/MASE · CRPS-sum · copertura{50,80,90} · ampiezza intervallo · tempo train/inferenza · n. parametri.
- **Supporta la tesi se:** M3 ≤ baseline su CRPS & calibrazione, anche se il MAE è solo comparabile.
- **Smentisce/complica se:** una seasonal-naive o DeepAR eguaglia M3 sul CRPS — *anch'esso un risultato di livello pubblicabile* ("la diffusione non ha pagato qui, ed ecco perché").
- **Artefatto:** la tabella-madre dei risultati + un grafico forecast-con-intervalli per modello.

### E2 — Sweep sull'orizzonte
- **Ipotesi:** il vantaggio della diffusione **cresce con l'orizzonte** (più futuro ⇒ più incertezza ⇒ i modelli generativi aiutano di più).
- **Setup:** `τ ∈ {corto, medio, lungo}` (es. 24/48/96), tutti i modelli.
- **Misura:** CRPS & copertura vs `τ`.
- **Artefatto:** grafico a linee CRPS-vs-orizzonte. Una storia monotona pulita è oro per una slide.

### E3 — Passi di denoising vs qualità & costo (l'esperimento specifico della diffusione)
- **Ipotesi:** c'è un **gomito** — la qualità si satura mentre il costo cresce linearmente in `T`.
- **Setup:** sweep `T ∈ {5,10,25,50,100,(250)}` su M3 (e/o sul DDPM giocattolo).
- **Misura:** CRPS vs `T` e tempo-di-inferenza vs `T` sugli stessi assi.
- **Perché conta:** rende operativo il limite dichiarato dalla survey (sampling iterativo lento) e dimostra che *capiamo* il meccanismo, non solo l'API.
- **Artefatto:** il grafico del compromesso qualità/costo — probabilmente la slide più memorabile.

### E4 — Robustezza / cambio di regime (l'extra del "perimetro bilanciato")
- **Ipotesi:** sotto distribuzione che cambia, la calibrazione si degrada; verifichiamo se la diffusione si degrada in modo **graduale** rispetto alle baseline.
- **Setup:** addestra su un periodo "normale", testa su uno spostato (stagione diversa / una rottura strutturale / un cluster di volatilità). La survey motiva esplicitamente questo (non-stazionarietà, cambi di regime).
- **Misura:** calo di CRPS & copertura, da train a shift; quale modello mantiene onesti gli intervalli.
- **Artefatto:** un grafico di calibrazione prima/dopo + una tabella di degrado.

### E5 — Generalità (facoltativo, Livello 2)
- **O** un secondo dataset (la storia di E1 regge?) **o** un secondo diffusion model (CSDI) **o** il contrasto con il GP sull'incertezza epistemica.
- **Artefatto:** una tabella "generalizza?" o la figura della scomposizione dell'incertezza.

---

## Parte 8 — Architettura software & best practice ingegneristiche

Il repository deve essere riproducibile, guidato da configurazioni e revisionabile da tre persone che programmano a livelli diversi.

### 8.1 Struttura del repository
```
pml-diffusion-tsf/
├── README.md                 # cosa/come eseguire, riassunto risultati
├── pyproject.toml / requirements.txt   # dipendenze fissate
├── configs/                  # un YAML per esperimento (modello+dati+seed)
│   ├── data_exchange.yaml
│   ├── model_deepar.yaml
│   └── exp_E1_main.yaml
├── src/
│   ├── data/                 # loader, splitting, scaling, finestratura, manifest
│   ├── models/               # wrapper sottili: naive, arima, deepar, timegrad, toy_ddpm
│   ├── eval/                 # metriche (CRPS, copertura, calibrazione), runner
│   ├── viz/                  # plotting coerente (forecast, intervalli, curve)
│   └── utils/                # seed, logging, timing, caricamento config
├── notebooks/                # esplorazione + il notebook didattico del toy-DDPM
├── experiments/              # script d'ingresso: run_E0.py ... run_E4.py
├── results/                  # CSV (il registro dei risultati) — committati
├── figures/                  # grafici generati per le slide — committati
└── runs/                     # checkpoint, log — esclusi da git (gitignore)
```

### 8.2 Ambiente & calcolo
- **Due ambienti:** uno locale e leggero (dati, baseline classiche, valutazione, grafici) e uno Colab/GPU (DeepAR, TimeGrad). Fissa le versioni — **la compatibilità PyTorchTS↔GluonTS è fragile**; registra la combinazione nota-buona in `requirements.txt` e nel README.
- **Disciplina Colab (il gruppo è nuovo):** monta Google Drive per dati/checkpoint; metti il blocco `pip install` nota-buono in cima a ogni notebook di addestramento; salva i checkpoint su Drive perché una disconnessione non perda un'esecuzione; tieni una cella "setup Colab" documentata una volta per tutte.

### 8.3 Config & riproducibilità
- Ogni esecuzione è completamente descritta da un **file YAML** (modello, dati, `H`, `τ`, `T`, seed). Nessun numero magico nel codice.
- **Seed globale** impostato per numpy/torch; loggato; flag deterministici attivati dove fattibile.
- Ogni esecuzione scrive una riga in `results/registry.csv` con hash della config + tutte le metriche + i tempi → la tabella dei risultati è *generata*, mai battuta a mano.

> **Sigle:** YAML (formato di configurazione leggibile dall'uomo); CSV (Comma-Separated Values, valori separati da virgola).

### 8.4 Tracciamento degli esperimenti
- Minimo: `results/registry.csv` strutturato + config salvate.
- Gradito: Weights & Biases (gratis) per le curve di loss — aiuta i membri nuovi al deep learning a *vedere* l'addestramento.

### 8.5 Usare bene Claude Code (conta, dato il mix di competenze)
Dagli compiti **piccoli, circoscritti, testabili**, non "costruisci il progetto". Buoni prompt:
- "Scrivi `src/data/load_exchange.py`: scarica, split temporale 70/10/20, mean-scaling solo-train, restituisci dataset GluonTS + manifest. Includi un `__main__` che stampa le shape."
- "Implementa `crps_sample(samples, target)` e un test unitario che lo confronta con un esempio a 3 punti calcolato a mano."
- "Dato un oggetto forecast di GluonTS, disegna mediana + bande 50/90% vs verità a terra; salva in `figures/`."
- "Scrivi `run_E3.py` che fa lo sweep di `T` e aggiunge righe CRPS + tempo-di-inferenza al registro."
Chiedigli sempre di aggiungere un piccolo test o un controllo di fumo `__main__`. Rivedi ogni diff a fronte delle trappole della Parte 6.6.

### 8.6 Checklist di code-review (eseguila prima di ogni merge)
☐ niente leakage di scaling fra split ☐ stessi `H,τ`,finestra-di-test fra i modelli ☐ seed loggato ☐ la metrica corrisponde alla definizione (dichiarato CRPS-sum vs medio) ☐ figura rigenerabile da uno script ☐ config committata.

---

## Parte 9 — Piano di gruppo: 3 passi isolati × 3 persone

Il lavoro è diviso in **tre passi fondamentali e isolati** (le unità-tempo del gruppo). Ogni passo ha un obiettivo, una *definizione di fatto*, dei prodotti e dei compiti per persona. I ruoli sono assegnati al profilo "a proprio agio con Python, nuovo al deep learning": distribuisci l'apprendimento del deep learning, dai a ciascuno una cosa che possiede pienamente per l'orale.

### Ruoli (ciascuno possiede una fetta verticale da capo a fondo, così ognuno sa rispondere all'orale)
- **Persona A — Responsabile Baseline & Statistica:** M0, M1, la narrativa classica/statistica-PML, le metriche puntuali.
- **Persona B — Responsabile Diffusione & Infrastruttura:** M2, M3, il DDPM giocattolo, Colab/addestramento, le config.
- **Persona C — Responsabile Valutazione, Visualizzazione & Storia:** CRPS/copertura/calibrazione, tutte le figure, la narrativa delle slide, il repertorio di domande orali.
Tutti leggono la Parte 2 (la spina dorsale) — l'orale valuta i singoli.

### Passo 1 — Capire, restringere, formalizzare *(chiudi quando sai dire il progetto in 30 secondi)*
- **Obiettivo:** trasformare survey + appunti in una domanda sperimentale precisa e congelata + una proposta inviata.
- **Definizione di fatto:** dataset scelto (primario + Exchange-per-iterare); `H`, `τ`, baseline, metriche fissati; riassunto di 1 pagina del paper/appunti scritto; proposta inviata via email (Appendice A); scheletro del repo + loader di Exchange esistono e girano.
- **Per persona:**
  - A: studia ARIMA/ETS + l'inquadramento del forecasting classico; abbozza le definizioni delle metriche (MAE/RMSE/MASE/CRPS/copertura).
  - B: studia §11.1–11.2 + il paper TimeGrad; allestisci lo scheletro del repo, gli ambienti e il loader di Exchange (preparazione-E).
  - C: studia la tassonomia della survey + §1.1.1/§9–10; abbozza la proposta e il riassunto di 1 pagina; predisponi l'impalcatura di figure/risultati.
- **Artefatti in uscita:** `PROPOSAL` inviata · `summary.md` · scheletro del repo · loader di Exchange verde.

### Passo 2 — Costruire la pipeline & eseguire gli esperimenti *(chiudi quando E0+E1 sono verdi sul dataset primario)*
- **Obiettivo:** produrre risultati quantitativi solidi.
- **Definizione di fatto:** E0 passa; tabella E1 completa con ≥3 seed; curva del compromesso E3 fatta; E4 tentato; tutte le figure generate da script.
- **Per persona:**
  - A: M0 + M1 funzionanti e valutati; tabelle delle metriche puntuali; figura diagnostica dei residui.
  - B: M2 (DeepAR) + M3 (TimeGrad) in addestramento su Colab; il DDPM giocattolo; lo sweep E3; checkpoint salvati.
  - C: il modulo di valutazione (CRPS/copertura/calibrazione) + il runner della tabella-madre; tutti i grafici di confronto/intervalli/calibrazione; tieni pulito il registro.
- **Artefatti in uscita:** `results/registry.csv` popolato · tabella-madre · grafici forecast/intervalli · curva E3 · (degrado E4).

### Passo 3 — Interpretare, scrivere la storia, preparare l'orale *(chiudi quando la presentazione di 8 minuti + il repertorio orale sono pronti)*
- **Obiettivo:** trasformare le tabelle in un argomento di 8 minuti e nella prontezza orale individuale.
- **Definizione di fatto:** 8–9 slide; note per chi parla; repertorio di domande&risposte orali per persona, mappato sulle pagine del corso; repo + README puliti; un elenco esplicito di "cosa abbiamo deliberatamente NON mostrato".
- **Per persona:**
  - A: le slide baseline/statistica + il punto di onestà "la baseline è davvero battuta?"; risposte orali ARIMA↔spazio-degli-stati.
  - B: la slide sul meccanismo della diffusione (forward/reverse/loss nella notazione degli appunti) + la storia del costo E3; risposte orali sulla diffusione (ricavare la loss).
  - C: l'arco narrativo, le slide risultati & calibrazione, la slide sulla scomposizione dell'incertezza (§2.3); prove + cronometraggio; assembla il repertorio orale.
- **Artefatti in uscita:** `slides.pdf` · `speaker_notes.md` · `oral_qa.md` · repo rifinito.

### 9.4 Livelli modulari "fermati-dopo" (perché la tempistica è flessibile)
| Livello | Contiene | È un progetto coerente da solo? |
|---|---|---|
| **Livello 0 — MVP** | Passi 1–2 parziali: E0 + E1 (M0–M3) su un dataset; slide di base | **Sì** — uno studio di confronto completo. |
| **Livello 1 — Obiettivo (il tuo "bilanciato")** | + E3 (passi vs costo) + E4 (cambio di regime) + calibrazione | Sì, e chiaramente forte per l'esame. |
| **Livello 2 — Allungo** | + E2 sweep completo, E5 (2° dataset / CSDI / GP), approfondimento del toy-DDPM, figura della scomposizione dell'incertezza | Sì, competitivo per la *lode*. |

> **Sigla:** MVP (Minimum Viable Product, prodotto minimo funzionante).

Punta prima al Livello 0; *poi* sali. Non lasciare mai il Livello 0 a metà per iniziare il Livello 2.

---

## Parte 10 — Registro dei rischi & mitigazioni
| Rischio | Probabilità | Impatto | Mitigazione | Responsabile |
|---|---|---|---|---|
| Inferno di versioni PyTorchTS↔GluonTS | Alta | Alto | fissa la combinazione nota-buona; isolala nel suo ambiente Colab; ripiego sul DDPM condizionato da zero (giocattolo ingrandito) | B |
| TimeGrad non converge / OOM | Media | Alto | sottoinsieme di canali; meno epoche/passi; batch più piccolo; prima Exchange | B |
| CRPS implementato male | Media | Alto | usa l'Evaluator di GluonTS + verifica su un giocattolo; cancello E0 | C |
| Leakage dei dati (scaling/finestra) | Media | Alto | loader centrale; checklist di review (8.6); manifest delle date di split | A/C |
| Sforamento di calcolo / € | Media | Medio | itera su Exchange; cache; limita i seed a 3; prima i modelli economici | B |
| Allargamento del perimetro (inseguire il Livello 2 presto) | Alta | Medio | imponi i cancelli di livello (9.4); MVP prima dell'allungo | tutti |
| L'orale espone una comprensione superficiale | Media | Alto | la spina dorsale (Parte 2) + il DDPM giocattolo + il repertorio orale per persona | tutti |
| Una persona diventa un collo di bottiglia | Media | Medio | fette verticali (ciascuno possiede un percorso eseguibile); sincronizzazione settimanale | tutti |

> **Sigla:** OOM (Out Of Memory, memoria esaurita).

---

## Parte 11 — Prodotti finali

### 11.1 La presentazione di 8 minuti (≈8–9 slide, un messaggio ciascuna)
1. **Problema** — "forecasting = futuri plausibili, non un numero." (il gancio)
2. **Inquadramento PML** — `p(y_futuro | x_passato)`; generativo + probabilistico; aleatoria vs epistemica.
3. **Idea della diffusione** — futuro vero → aggiungi rumore → impara a denoisare *condizionando sul passato* → campiona traiettorie. (uno schema pulito)
4. **Dove si colloca nel corso / nella survey** — §11.2 resa condizionata; tassonomia della survey (fonte di condizionamento + integrazione) in una riga.
5. **Setup sperimentale** — dataset, `H`, `τ`, split temporale, la scala dei modelli.
6. **Risultato principale (E1)** — la tabella: MAE/RMSE · CRPS · copertura · costo. Più un grafico forecast-con-intervalli.
7. **L'intuizione specifica della diffusione (E3, +E4)** — compromesso qualità-vs-passi-di-denoising; (calibrazione al cambio di regime).
8. **Discussione** — quando la diffusione vince / perde; quale incertezza catturiamo e ci sfugge; la riserva sul costo.
9. **Conclusione** — "incertezza più ricca e meglio calibrata, a un costo di sampling regolabile; utile quando l'incertezza conta." + lavori futuri (diffusione bayesiana/a ensemble).
- **Deliberatamente NON mostrato:** ogni ablazione, ogni seed, l'idraulica delle librerie, le esecuzioni fallite. Tieni solo ciò che sostiene la tesi.

### 11.2 Artefatti di supporto
- `speaker_notes.md` (1 pagina), il repo + README + `results/registry.csv` + `figures/`.

### 11.3 Preparazione dell'esame orale (per persona, mappata sulle pagine del corso)
Costruisci `oral_qa.md` con risposte che ogni membro sa dare. Domande-seme:
- *Ricava la loss del DDPM; perché predire ε e non x₀?* → §11.2.3 (p.126). (guida B)
- *Perché il processo reverse è ~gaussiano? quando questo fallisce?* → §11.2.2 (p.125). (B)
- *Dove sono l'ELBO / il reparameterization trick qui?* → §9.2 (p.96), §10.4.1 (p.112). (B/C)
- *Cos'è il CRPS, perché è "proprio", come si rapporta al MAE?* → §2.3 (p.20). (C)
- *Aleatoria vs epistemica — quale cattura la tua diffusione?* → §1.1.1 (p.2). (C)
- *Come produce DeepAR una distribuzione? vs diffusione?* → §1.2 (p.3). (A/B)
- *ARIMA come modello probabilistico — assunzioni?* → Cap. 5/6. (A)
- *Vista score-based & legame con Langevin/MCMC?* → §11.2.5 (p.127), Cap. 8. (B)
- *Perché il sampling della diffusione è lento; come l'hai misurato/mitigato?* → E3. (B/C)

---

## Parte 12 — Piano di studio (cosa leggere, quando, chi)
Ricorda che il gruppo ha studiato fino al Cap. 5. Il progetto richiede soprattutto i Cap. 9–11.
| Da leggere | Chi | Quando (passo) | Perché |
|---|---|---|---|
| §9.2 ELBO, §10.4.1 reparam., §10.6 BNN | tutti | Passo 1 | il macchinario che la diffusione riusa |
| §11.1 VAE, §11.2 Diffusione (tutto) | tutti (B più a fondo) | Passo 1 | il cuore; B deve saperlo ricavare |
| Survey *Diffusion Models for TSF* (2507.14507) | C (+tutti in scorsa) | Passo 1 | tassonomia, inquadramento, lavori correlati |
| Paper TimeGrad (Rasul et al., ICML'21) | B | Passo 1–2 | il modello che eseguiamo |
| Paper DeepAR (Salinas et al.) | A/B | Passo 2 | la baseline di deep learning |
| CRPS / scoring proprio (Gneiting & Raftery '07) | C | Passo 1–2 | correttezza della metrica |
| §1.1.1 incertezza; Cap. 12 GP (in scorsa) | C | Passo 2–3 | la slide di discussione |
| (facolt.) paper CSDI | B | Passo 3 (Livello 2) | secondo diffusion model |

> **Sigla:** TSF (Time Series Forecasting, forecasting di serie temporali).

---

## Appendice A — Proposta pronta da inviare (copia-incolla / da inviare al professore)

> **Oggetto:** Proposta di progetto — Forecasting Probabilistico di Serie Temporali con Modelli di Diffusione
>
> Gentile Professor Bortolussi,
>
> il nostro gruppo (3 studenti) propone un progetto sul **forecasting probabilistico di serie temporali con modelli di diffusione**, estendendo il materiale sulla diffusione del Capitolo 11 dalla generazione non condizionata al forecasting *condizionato*.
>
> **Domanda.** Un diffusion model condizionato può stimare `p(futuro | passato)` — una distribuzione di traiettorie future plausibili — con un'incertezza meglio calibrata rispetto alle baseline classiche e probabilistiche di deep learning, e a quale costo di sampling?
>
> **Metodo.** Su un dataset reale multivariato (es. Solar/Electricity) confrontiamo una baseline seasonal-naive, un modello classico (ARIMA/ETS), un modello probabilistico autoregressivo di deep learning (DeepAR) e un diffusion model (TimeGrad), più un piccolo DDPM condizionato scritto da zero a partire dagli Algoritmi della §11.2 per dimostrare la comprensione. Valutiamo l'accuratezza puntuale (MAE/RMSE/MASE), la **qualità probabilistica (CRPS, copertura, calibrazione)** e il **costo computazionale** (incluso un compromesso qualità-vs-passi-di-denoising e un test di robustezza al cambio di regime).
>
> **Legame con il corso.** L'obiettivo di addestramento è la loss di predizione del rumore della §11.2 (un ELBO ri-parametrizzato, §9.2/§10.4.1); il sampling è ancestral sampling sulla catena inversa (§3.3.1); discutiamo anche la vista score-based/Langevin (§11.2.5) e che *tipo* di incertezza — aleatoria vs epistemica (§1.1.1) — il modello cattura.
>
> Questo perimetro sarebbe adatto per il progetto? Siamo lieti di modificare il dataset o l'insieme delle baseline.
>
> Cordiali saluti, [nomi]

## Appendice B — Glossario (termine → in parole semplici → pagina del corso)
- **Incertezza aleatoria / epistemica** — rumore irriducibile dei dati / ignoranza riducibile del modello — §1.1.1 (p.2).
- **ELBO** — il limite inferiore che massimizziamo al posto della verosimiglianza intrattabile — §9.2 (p.96).
- **Reparameterization trick** — scrivere `z=μ+σ·ε` perché i gradienti attraversino il sampling — §10.4.1 (p.112).
- **Diffusione forward/reverse** — catena di rumore fissa / catena di denoising appresa — §11.2.1–2 (p.123).
- **Loss di predizione del rumore** — addestrare `ε_θ` a predire il rumore aggiunto — §11.2.3 (p.126).
- **CRPS** — generalizzazione distribuzionale del MAE; una regola di scoring propria — §2.3 (p.20).
- **Copertura / calibrazione** — gli intervalli al x% contengono la verità il x% delle volte — valutazione (Parte 6).
- **CRPS-sum** — il CRPS multivariato usato da TimeGrad (somma i canali, poi calcola).
- **Pinball (quantile) loss** — la perdita asimmetrica il cui minimizzatore è un dato quantile; il CRPS = la sua media su tutti i livelli di quantile — §2.3 (p.20).

## Appendice C — Riferimenti
- *Diffusion Models for Time Series Forecasting: A Survey*, arXiv:2507.14507 (2025).
- Rasul et al., *Autoregressive Denoising Diffusion Models for Multivariate Probabilistic TSF* (TimeGrad), ICML 2021.
- Tashiro et al., *CSDI*, NeurIPS 2021.
- Salinas et al., *DeepAR*, Int. J. Forecasting 2020.
- Ho et al., *Denoising Diffusion Probabilistic Models* (DDPM), NeurIPS 2020.
- Gneiting & Raftery, *Strictly Proper Scoring Rules*, JASA 2007.
- Gneiting & Katzfuss, *Probabilistic Forecasting*, Annual Review of Statistics 2014 — calibrazione e scoring proprio dei forecast probabilistici.
- Bortolussi, *Appunti del corso di PML* — Cap. 1, 2, 3, 5, 8, 9, 10, 11, 12.
- Yang et al., *Diffusion Models: A Comprehensive Survey* (2022) — già nella cartella PML.

## Appendice D — Schede-dati rapide del dataset (da compilare durante il Passo 1)
| | primario = ? | iterazione = Exchange | allungo = ? |
|---|---|---|---|
| D (canali) | | 8 | |
| frequenza | | giornaliera | |
| lunghezza | | ~6k | |
| H / τ scelti | | / | |
| idea di cambio-regime (E4) | | | |
| CRPS-sum pubblicato (target E0) | | | |

---

## Appendice E — La fase sperimentale, spiegata a fondo (metodologia + runbook operativo)

Questa appendice esiste perché la **fase sperimentale è il cuore valutabile del progetto**. La Parte 7 elenca *quali* esperimenti facciamo; qui spieghiamo **prima il "perché" (la metodologia)** e **poi il "come" passo-passo (il runbook)**. Leggi la prima metà per capire, la seconda per eseguire.

### E.1 — Metodologia: cosa rende un esperimento valido

#### E.1.1 Un esperimento non è "far girare il codice": è un test che può fallire
Nel nostro progetto un esperimento ha cinque parti obbligatorie (le stesse della Parte 7):
1. **Ipotesi** — una previsione precisa e *falsificabile* ("M3 ha CRPS più basso di DeepAR sul dataset primario").
2. **Setup** — l'ambiente controllato: dataset, `H`, `τ`, seed, modelli coinvolti.
3. **Cosa varia / cosa si misura** — la *variabile indipendente* che muoviamo (es. `T`) e la *variabile dipendente* che osserviamo (es. CRPS, tempo).
4. **Esito che supporta vs smentisce** — deciso *prima* di guardare i numeri. Se non sai dire in anticipo quale risultato ti smentirebbe, non è un esperimento, è una demo.
5. **Artefatto** — la tabella o il grafico che resta come prova (e che finirà su una slide).

La regola mentale: **un buon esperimento può deludere la nostra tesi, ed è lo stesso un risultato.** Se la diffusione perde su un dataset, lo riportiamo e spieghiamo perché — questo *aumenta* il voto (onestà, comprensione), non lo abbassa.

#### E.1.2 Il principio del confronto equo (è ciò che rende i numeri credibili)
Tutto il valore del confronto E1 dipende dal fatto che ogni modello giochi *con le stesse regole*. Concretamente:
- **Stesso "contratto sui dati"** (Parte 4.5): identici split, scaling, `H`, `τ`, finestre di test per tutti i modelli. Un solo loader li serve tutti.
- **Stesso modo di estrarre il punto e l'intervallo** (Parte 3.4): la metrica puntuale si calcola sempre dalla media/mediana predittiva, anche per i modelli probabilistici.
- **Nessuna messa a punto asimmetrica:** non confrontare un TimeGrad ottimizzato con un DeepAR lasciato ai default. O metti a punto entrambi, o nessuno, e lo dichiari.
- **Niente scelte a posteriori:** l'orizzonte, il seed e la soglia si fissano *prima*. Scegliere dopo il più favorevole è cherry-picking (Parte 6.6).

#### E.1.3 Il cancello di riproduzione (E0): perché si parte da lì
Prima di fidarsi di *qualunque* numero prodotto da noi, riproduciamo un numero **già pubblicato** (un CRPS-sum di TimeGrad o DeepAR su un dataset noto) entro una tolleranza ragionevole (stesso ordine di grandezza, ≲20%). Logica:
- Se riproduciamo un risultato noto, la pipeline (caricamento, scaling, metrica) è probabilmente corretta.
- Se **non** ci riusciamo, c'è un bug a monte (leakage, scaling, CRPS sbagliato) e *tutti* i numeri successivi sarebbero spazzatura.
- È anche il punto in cui impari a usare gli strumenti su un bersaglio di cui conosci già la risposta — il modo meno frustrante di sbagliare.

**Finché E0 non è verde, nessun altro esperimento conta.** È letteralmente un cancello.

#### E.1.4 Igiene statistica: il rumore della casualità
I modelli appresi dipendono dal seed casuale (inizializzazione dei pesi, ordine dei batch). Un singolo numero può essere fortunato o sfortunato. Perciò:
- Esegui ogni modello appreso con **≥3 seed** e riporta **media ± deviazione standard**.
- Se due modelli distano meno di una deviazione standard, *non* dichiarare un vincitore: di' "indistinguibili su questi dati".
- Tieni i seed fissi e loggati, così chiunque può ri-eseguire e ottenere gli stessi numeri.

#### E.1.5 Come si legge l'output (cosa "ha un bell'aspetto")
- **CRPS / CRPS-sum:** più basso è meglio. Da solo non dice nulla: ha senso solo *relativo* (vs baseline) o *vs il numero pubblicato* (E0).
- **Copertura @ 90%:** vuoi ~90%. Molto sotto = il modello è troppo sicuro (intervalli stretti e bugiardi); molto sopra = troppo prudente (intervalli larghi e inutili).
- **Ampiezza dell'intervallo:** a parità di copertura, più stretto è meglio (più "affilato").
- **Istogramma PIT:** vuoi che sia **piatto**. A "U" = troppo sicuro; a campana = troppo prudente. È il controllo visivo di onestà del forecast.
- **Curva CRPS-vs-`T` (E3):** cerchi un **gomito** — un punto oltre il quale aggiungere passi di denoising non migliora più la qualità ma continua a costare tempo. Quel gomito è la "slide memorabile".

#### E.1.6 Minacce alla validità (la lista da temere)
Sono i modi in cui un esperimento può *sembrare* riuscito ed essere falso. Stampatela e usatela come checklist:
- **Look-ahead leakage:** il modello vede informazione futura (es. una covariata calcolata sull'intera serie).
- **Scaling leakage:** lo scaler è stato addestrato includendo val/test.
- **Finestra a cavallo:** una finestra `(H, τ)` attraversa il confine train/test.
- **Confronto iniquo:** modello distribuzionale vs puntuale senza riportarli entrambi alla stessa metrica.
- **Cherry-picking:** scelta a posteriori di seed/orizzonte/soglia favorevoli.
- **Ambiguità del CRPS:** riportare "CRPS" senza dire se è per-canale medio o CRPS-sum (non sono confrontabili).

### E.2 — Runbook: come si esegue davvero, passo per passo

> Presuppone lo scheletro di repo della Parte 8.1 e i due ambienti della Parte 8.2. Ordine d'oro: **prima fai girare tutto su Exchange** (minuti, anche senza GPU), *poi* passi al dataset primario su Colab.

#### E.2.0 Preparazione dell'ambiente (una volta)
1. **Locale (leggero):** crea un ambiente virtuale; installa il gruppo "leggero" (numpy, pandas, Darts/statsforecast, le metriche, matplotlib). Qui girano dati, baseline classiche, valutazione e grafici.
2. **Colab (GPU):** in un notebook, prima cella = il blocco `pip install` nota-buono (versioni *fissate* di PyTorchTS + GluonTS + torch compatibili — la compatibilità è fragile, Parte 8.2). Seconda cella = monta Google Drive. Terza cella = clona/aggiorna il repo. Salva i checkpoint su Drive, non sul disco effimero di Colab.
3. **Verifica di fumo:** esegui il `__main__` del loader di Exchange; deve stampare le shape di train/val/test senza errori. Se questo non gira, non procedere.

#### E.2.1 Eseguire E0 (il cancello) — fallo per primo
1. Scegli la coppia (modello, dataset) con un CRPS-sum **pubblicato** (es. TimeGrad su Exchange o Electricity). Annota il numero-bersaglio nell'Appendice D.
2. Scrivi/usa `experiments/run_E0.py`: carica i dati col loader centrale, addestra il modello con la config (YAML), campiona `S` traiettorie sul test, calcola il CRPS-sum col modulo di valutazione.
3. Confronta col numero pubblicato.
   - **Entro tolleranza?** → E0 verde. Scrivi la riga di "riproduzione" nel registro e una nota sugli scarti. Procedi.
   - **Fuori tolleranza?** → *non procedere*. Controlla, in quest'ordine: definizione del CRPS (sum vs medio?), leakage di scaling, allineamento di `H/τ`, numero di campioni `S`. Itera finché non rientra.
4. Verifica incrociata del CRPS: calcolalo a mano su un esempio a 3 punti e confrontalo con il modulo, una volta, per fidarti del codice.

#### E.2.2 Eseguire E1 (il confronto principale)
1. Fissa, *prima*, `H`, `τ`, il dataset primario, l'insieme dei seed (≥3).
2. Per ciascun modello `M0, M1, M2, M3` e per ciascun seed: esegui via la sua config YAML; ogni esecuzione **aggiunge una riga** a `results/registry.csv` con hash-config + tutte le metriche + i tempi. Niente numeri battuti a mano.
3. Genera la tabella-madre *da* `registry.csv` (uno script, non copia-incolla): per ogni modello, media ± deviazione standard sui seed.
4. Genera un grafico forecast-con-intervalli per modello (mediana + bande 50/90% vs verità a terra).
5. Leggi i risultati con la testa della §E.1.5; scrivi una frase di conclusione *prima* di abbellire le slide.

#### E.2.3 Eseguire E3 (passi di denoising vs qualità/costo)
1. Tieni tutto fisso tranne `T`; definisci la griglia `T ∈ {5,10,25,50,100,(250)}`.
2. `experiments/run_E3.py` cicla su `T`: per ciascun valore, campiona dal modello già addestrato (non ri-addestrare!), registra CRPS *e* tempo-di-inferenza per finestra.
3. Disegna due curve sugli stessi assi (CRPS vs `T`; tempo vs `T`). Cerca il gomito (§E.1.5).
4. Se usi anche il DDPM giocattolo, ripeti: è qui che torna utile, perché lo controlli completamente.

#### E.2.4 Eseguire E4 (robustezza al cambio di regime)
1. Definisci due periodi: "normale" (train) e "spostato" (test) — una stagione diversa, una rottura strutturale, un cluster di volatilità. Documenta il criterio.
2. Addestra sul normale; valuta sia sul normale sia sullo spostato.
3. Misura il *calo* di CRPS e copertura passando normale→spostato, per ogni modello.
4. Artefatto: un grafico di calibrazione prima/dopo + una tabella di degrado. La storia è "chi mantiene onesti gli intervalli sotto shift?".

#### E.2.5 Disciplina trasversale (vale per ogni esperimento)
- **Una config = una esecuzione:** nessun numero magico nel codice; tutto in YAML (modello, dati, `H`, `τ`, `T`, seed).
- **Checkpoint salvati su Drive** dopo ogni addestramento, così una disconnessione di Colab non costa un'esecuzione.
- **Il registro è la verità:** la tabella delle slide si *rigenera* da `results/registry.csv`. Se un numero non è nel registro, non esiste.
- **Le figure si rigenerano da script** (Parte 8.6): mai una figura "fatta a mano" che non sai ricostruire.
- **Checklist pre-merge (Parte 8.6)** a ogni passo: niente leakage, stessi `H/τ`/finestra, seed loggato, metrica dichiarata, figura rigenerabile, config committata.

#### E.2.6 "Definizione di fatto" della fase sperimentale
La fase sperimentale è chiusa (Livello 1, il "bilanciato") quando:
- ☐ E0 è verde e documentato (riga di riproduzione + nota scarti);
- ☐ la tabella E1 è completa, con media ± deviazione standard su ≥3 seed, generata da script;
- ☐ esiste la curva del compromesso E3 con un gomito leggibile;
- ☐ E4 è stato tentato, con grafico di calibrazione prima/dopo;
- ☐ ogni numero in ogni figura è tracciabile fino a una riga di `results/registry.csv` e a una config YAML committata.
