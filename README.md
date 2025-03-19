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