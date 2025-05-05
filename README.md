# info381
xAI prosjekt

## Oppgave beskrivelse

Utforske hvordan en kan bruke regel-uthenting på maskinlæringsmodeller for svindel-gjenkjenning og hvorvidt de kan brukes til å håndheve regelverk. 

## Modeller

- RuleFit
- XGBoost
- Random Forest 
- Autoencoder

## Metoder

- Global shap for å finne hvilke features som har mest å si globalt for modellen
- LORE
- Anchor
- DiCE

## Datasets

- [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data )

## Oppgave krav

**Deadline**: May 8th, 16:00

**Format**: Written report, up to 6000 words. Include links to code
repository, runnable code and data sets.

**Group size**: 3 people 

**Task**:

The goal of this task is to make experimental software solutions that use XAI software to create explanations for one or several self-selected datasets. The group needs to either

- Identify 3-5 machine learning methods with XAI approaches for one single dataset. A significant variation in the approaches chosen is expected. A prototype that enables a user to test out the XAI methods is expected.

- Identify 3-5 data sets and apply one single adapted XAI approach to all the datasets. The data sets should have different properties in terms of number of data points, number of features, feature types etc. A prototype that enables a user to test out the method application on the different data sets is expected.

Data sets can be found at kaggle.com. Software for machine learning and XAI can be found in many places. Molnar’s textbook mentions resources for XAI. One can also find links to useful software at this link: https://tdan.com/explainable-ai-5-open-source-tools-you-should-know/31589

The report (in English or Norwegian) shall contain
1) Title and authors

2) A short abstract (150 words) describing the work and a summary of results.

3) A problem description with motivation leading to research questions like: How does method Y work on diverse data sets? Or Which XAI methods works well with data set Y (and similar data)?

4) Description of the data set(s)

5) The XAI method(s) chosen

6) Set up for evaluation (experiments)

7) Quantitative and qualitative results with visualizations (if possible).

8) Discussion and conclusions

9) Links to code repository, runnable code and data sets.

10) Reference list

Grading counts 60 % of total grade and will be based on
- Complexity of chosen data sets and XAI methods
- Quality of the software developed
- Experimental setup
- Analysis of results and discussion
- Report writing, organization and readability

# SkopeRules instrukser

Aktiver conda med `source ~/anaconda3/bin/activate` først.

I terminal:

1. kjør `conda create -n skoperules python=3.5`
2. kjør `conda activate skoperules`
3. kjør `pip install skope-rules matplotlib`
4. kjør `pip freeze` for å sjekke pakkeversjoner

**NB**: Ved å installere skope-rules, burde avhengigetene installeres automatisk. Versjonene av pakkene burde være:

```txt
Python (>= 2.7 or >= 3.3)
NumPy (>= 1.10.4)
SciPy (>= 0.17.0)
Pandas (>= 0.18.1)
Scikit-Learn (>= 0.17.1)
Matplotlib >= 1.1.1 is required.
```

**NBB**: husk å velge skoperules som interpreter i VSCode

1. i VSCode, trykk ctrl+shift+P
2. Skriv 'interpreter'
3. velg "Select Python Interpreter"
4. velg skoperules

## Kilder

- [SkopeRules docs](https://skope-rules.readthedocs.io/en/latest/api.html)
- [SkopeRules github](https://github.com/scikit-learn-contrib/skope-rules)


# To do: 

1. Globale shapley verdier + regelsett + classification report fra nå.
- [ ] Globale Shapley-verdier.
- [ ] Regelsett.
- [ ] Classification report.
2. Globale shapley verdier + regelsett + classification report etter at vi har skalert ned til 10 features og fjernet skalering.
- [ ] Globale Shapley verdier.
- [ ] Regelsett.
- [ ] Classification report.
3. Globale shapley verdier + regelsett + classification report etter at vi har redusert til 5 features og fjernet så mye preprosessering som mulig, så tallene som kommer ut er leselige.
- [ ] Globale Shapley verdier.
- [ ] Regelsett.
- [ ] Classification report.
4. Prototype - se beskrivelsen Bjørnar har lagt ut. 
4. Prototype - se beskrivelsen Bjørnar har lagt ut.
- [ ] 2-3 eksempler som viser regler og shapley verdier
- [ ] Interaktivt script

### Anchor resultater fra nn advanced

Most common anchor rules:
('amt <= 0.17', 'trans_hour_cos <= -0.35'): 193 times
('amt <= 0.17', 'trans_hour_cos <= 1.02'): 138 times
('amt <= 2.74', 'trans_hour_cos <= 1.02'): 28 times
('amt <= 2.74', 'trans_hour_cos <= -0.35'): 21 times
('amt <= 0.17', 'trans_hour_sin <= -0.52'): 7 times
('amt <= 0.17', 'trans_hour_sin <= -0.52', 'category <= 0.20'): 6 times
('category > 1.12', 'trans_hour_cos <= -0.35'): 5 times
('category > 1.12', 'amt <= 2.74'): 5 times
('category > 0.20', 'trans_hour_cos <= -0.35'): 5 times
('amt <= 0.17', 'trans_minute_sin > 0.94', 'trans_dayofweek <= -0.90'): 4 times