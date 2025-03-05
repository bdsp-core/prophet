# Design Document

## EHRPredict
`EHRPredict` is the main class that provides access to the individual models. The hope is to make this as clean and simple as possible, being able to do everything necessary in a single line of code on the user's end.

```python
from prophet import EHRPredict
import pandas as pd

data = {
    'icd_codes': pd.read_csv('icd_codes.csv'),
    'medications': pd.read_csv('medications.csv'),
    'notes': pd.read_csv('notes.csv')
}

model = EHRPredict()
results = model.run('congestive heart failure', data=data)

# this would also work
# model.run('congestive heart failure', icd_codes=data['icd_codes'], medications=data['medications'], notes=data['notes'])
```

`EHRPredict` creates individual instances of the models via the `_ModelCreator` class. It then has two methods (so far) that can be called: `run()` and `get_data_format()`.

`get_data_format()` is a static method that returns the data format that the phenotype model expects as a dictionary, with the keys being the names of the dataframes and the values being the column names. More on this later.

```python
EHRPredict.get_data_format('congestive heart failure')
# {
#     "icd_codes": [
#         "date_icd",
#         "icd",
#         "bdsp_patient_id"
#     ],
#     "medications": [
#         "date_med_start",
#         "date_med_end",
#         "med",
#         "bdsp_patient_id"
#     ],
#     "notes": [
#         "date_note",
#         "note",
#         "bdsp_patient_id"
#     ]
# }
```

`run()` is the end all be all. It takes the name of the phenotype and the data dictionary as input and returns the identity dataframe, feature matrix dataframe, and the predictions.

## Your individual model

Your model should inherit from the `_BaseModel` class and implement its abstract methods. This includes:

- `get_data_format()`
- `preprocess_data()`
- `generate_features()`
- `predict()`

and anything else if we decide to change things. Have your model be all loaded and ready to go when it is initialized `__init__()` This includes stuff like downloading nltk datasets, loading up your vocabulary, etc.

### `get_data_format()`

This is a class method (does not need the class to be initialized to be called) that should return a dictionary with the keys being the names of the dataframes and the values being the column names. This is mostly for the user **HOWEVER** it would be **VERY NICE** if we could keep things consistent across the models. I.e., have columns/dfs be named consistently, contain the same information, etc. We can talk about this and figure out what works best for everyone.

### `preprocess_data()`

This is **STEP 1**. Input data will be a dictionary with the keys being the names of the dataframes and the values being the dataframes <mark style="background: #dee8a5">and/or just dfs be input as named kwargs?</mark>. This **should** be input in the same way as `get_data_format()` **BUT** things are messy so it would be great to error check/clean up anything not needed. Things that you should do here:

- Check that data is in the correct format, removing any unnecessary or empty columns/rows, and converting to proper types (datetime, int, str).
- Reset the index of each dataframe, just in case.
- Throw an error if you don't have something you need.

The output of this will be fed into your `generate_features()` method <mark style="background: #dee8a5">(and saved with pickle? pandas? if save_dir is passed into `run()`)?</mark>

### `generate_features()`

**STEP 2** just do your special magic and make the features from the data. Return two things: the identity dataframe and the feature matrix dataframe. The division is just so we can feed the features right into the model, but keep the identifiers for the output for later.

IMHO the identity dataframe should be roughly the same format for all models. At the very least, each row = a patient's note and bdsp_patient_id, which corresponds to the feature matrix row. Bonus points for including ICD +/- column, medication +/-, etc.

### `predict()`

**STEP 3** Should be as easy as running `.predict()` and `.predict_proba()` on your model. Returns a dataframe with the ouptput of the model. Name the columns appropriately.

This will return the identity dataframe, the feature matrix dataframe, and the predictions. So don't tweak the row order or change the length of the dataframes please.

### Other things

I'm thinking the file structure should roughly go like this:

  ```python
  prophet
  |--- models
  |    |--- your_model
  |    |    |--- code           # your functions, etc
  |    |    |    |--- module.py 
  |    |    |--- files          # whatever files you read from for your model
  |    |    |    |--- vocab.txt 
  |    |    |--- test           # test files, .gitignore'd
  |    |    |    |--- test_data.csv
  |    |    |--- __init__.py    # if you need it
  |    |    |--- _yourmodel.py  # class
  ...
  ```  

It's super annoying to try to have different versions of python libraries for different model environments. If necessary we can create a Docker container to run models, but would be a billion times easier to just use the same versions of stuff across the board.

## Model Status

<span style='color: #e64747;'>Model not available yet</span>  
<span style='color: #e09c3b'>Model finished</span>  
<span style='color: #e6e22e'>Model integrated</span>  
<span style='color: #8fb935'>Model run on data **[COHORT]**</span>

### NLP models

- <span style='color: #e6e22e'>Congestive Heart Failure</span>
- <span style='color: #e09c3b'>Epilepsy [Marta's]</span>
- <span style='color: #e09c3b'>Epilepsy [Carolina's]</span>
- <span style='color: #e09c3b'>Ischemic Stroke</span>
- <span style='color: #e09c3b'>Intracranial Hemorrhage</span>
- <span style='color: #e09c3b'>Mild Cognitive Impairment / Alzheimer's Disease</span>
- <span style='color: #e09c3b;'>Subarachnoid Hemorrhage</span>
- <span style='color: #8fb935;'>Parkinson's Disease: **MGB**</span>
- <span style='color: #e09c3b;'>Cardiac Arrest</span>
- <span style='color: #e09c3b;'>Traumatic Brain Injury</span>
- <span style='color: #e09c3b;'>Brain Tumors</span>
- <span style='color: #e09c3b;'>Subdural Hematoma</span>
- <span style='color: #e64747;'>Cefepime Neurotoxicity</span>

### Rule-based models

- <span style='color: #e09c3b'>Myocardial Infarction</span>
- <span style='color: #e09c3b'>Ischemic Stroke</span>
- <span style='color: #e09c3b'>Atrial Fibrillation</span>
- <span style='color: #e09c3b'>Coronary Artery Disease</span>
- <span style='color: #e09c3b'>Diabetes Mellitus</span>

## ToDos / Fixes / Feature Requests

- implement multiprocessing for feature matrix generation
- singleton versions of models, or at least a way to not have to reinitialize the model for each run
- save intermediate steps
- batch processing for large amounts of data
