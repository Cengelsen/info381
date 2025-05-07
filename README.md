# INFO 381: Research Topics in Artificial Intelligence

This repository contains the code used for a semester project in a master subject at the University of Bergen. 

## Instructions for installing SkopeRules:

Activate conda with `source ~/anaconda3/bin/activate`.

In terminal:

1. run `conda create -n skoperules python=3.5`
2. run `conda activate skoperules`
3. run `pip install skope-rules matplotlib`
4. run `pip freeze` to check package versions

**NB**: By installing skope-rules, the dependencies should be automatically installed. The versions of the packages should be:

```txt
Python (>= 2.7 or >= 3.3)
NumPy (>= 1.10.4)
SciPy (>= 0.17.0)
Pandas (>= 0.18.1)
Scikit-Learn (>= 0.17.1)
Matplotlib >= 1.1.1 is required.
```

**NBB**: Remember to select the conda environment as interpreter in VSCode

1. in VSCode, press ctrl+shift+P
2. Write 'interpreter'
3. choose "Select Python Interpreter"
4. choose skoperules

## Sauces

- [SkopeRules docs](https://skope-rules.readthedocs.io/en/latest/api.html)
- [SkopeRules github](https://github.com/scikit-learn-contrib/skope-rules)
