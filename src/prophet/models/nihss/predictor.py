from ...base._basemodel import _BaseModel
import logging
from typing import Dict, Union, Tuple
import polars as pl
import re
import ray
import joblib
import time
import gc
import shutil
import tempfile
import psutil
from pathlib import Path
from datetime import datetime
import importlib.resources
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# 266 binary n-gram features (stemmed) in training order
NIHSS_FEATURES = [
    'alert', 'aphasia', 'approxim', 'aspir', 'aspirin', 'bedsid', 'brought', 'cathet',
    'comfort', 'cultur', 'deni', 'deviat', 'dizzi', 'drift', 'dysarthria', 'em', 'endors',
    'esr', 'feed', 'feel', 'felt', 'foley', 'function', 'gaze', 'headach', 'iat', 'intub',
    'light', 'long', 'make', 'maximum', 'mca', 'membran', 'movement', 'nausea', 'none',
    'numb', 'nurs', 'occup', 'old', 'ongo', 'palsi', 'paralysi', 'perfus', 'plavix', 'poor',
    'prefer', 'proxim', 'resolv', 'respond', 'respons', 'shift', 'slightli', 'small', 'soft',
    'speech', 'spontan', 'strength', 'subsequ', 'taper', 'tia', 'tingl', 'trauma', 'unabl',
    'venou', 'vertebr', 'vision', 'work', 'xr', 'activ toler', 'all extrem', 'all muscl',
    'ani histori', 'ani medic', 'arm left', 'awar touch', 'bedsid statu', 'behavior neg',
    'bolu intraven', 'bp control', 'bp devic', 'brain without', 'care intervent', 'cathet less',
    'cerebellar infarct', 'cerebr infarct', 'clinic improv', 'cn intact', 'command follow',
    'command gaze', 'commun appear', 'constitut orient', 'continu home', 'ct chest', 'deni ani',
    'diet npo', 'diet regular', 'distal right', 'drift strength', 'dysarthria normal', 'ed no',
    'effort graviti', 'electrocardiogram normal', 'electrocardiogram rhythm', 'esr crp',
    'extrem not', 'extrem pain', 'extrem reflex', 'felt like', 'flexion left', 'focal weak',
    'function baselin', 'gait stair', 'gaze deviat', 'gaze prefer', 'given age', 'given oral',
    'home none', 'inattent extinct', 'infarct left', 'infus minimum', 'infus stop', 'intact cn',
    'intact strength', 'intraven tpa', 'left leg', 'left mca', 'left motor', 'left normal',
    'left side', 'less drain', 'long term', 'lower facial', 'mca stroke', 'mild left',
    'mild narrow', 'motor left', 'move left', 'move right', 'mra head', 'mri angiographi',
    'musculoskelet no', 'neck cta', 'neck pain', 'neck vessel', 'net urin', 'neuro alert',
    'neuro consult', 'no bleed', 'no cough', 'no drift', 'no dysarthria', 'no effort',
    'no eye', 'no left', 'no mgr', 'no movement', 'no pronat', 'no signific', 'no visual',
    'normal strength', 'not move', 'numb weak', 'observ statu', 'obtain due', 'open eye',
    'pariet lobe', 'partial paralysi', 'perform not', 'perrl eomi', 'plan mdm', 'plan right',
    'pulmonari no', 'ra none', 'rang minimum', 'report no', 'result clinic', 'review notabl',
    'right lower', 'right side', 'round reactiv', 'sensat intact', 'sever dysarthria',
    'side weak', 'singl no', 'site right', 'smile tongu', 'speech clear', 'speech slur',
    'stabl no', 'statu chang', 'statu prior', 'stroke call', 'stroke mild', 'stroke past',
    'stroke tia', 'stroke workup', 'subacut infarct', 'symptom improv', 'symptom resolv',
    'telemetri laboratori', 'territori infarct', 'tongu midlin', 'unabl assess', 'unabl obtain',
    'unabl perform', 'unabl test', 'urin cultur', 'urinari cathet', 'visual complet', 'woke up',
    'activ ppx gastrointestin', 'alcohol not sh', 'all muscl perform', 'all review neg',
    'answer not question', 'bedsid statu full', 'bp devic none', 'bulk tone normal',
    'care progress femal', 'cn visual field', 'consult constip prophylaxi', 'ed cours condit',
    'facial palsi normal', 'flow cardiovascular rrr', 'follow complex command', 'form complet not',
    'graviti motor leg', 'histori mental statu', 'histori physic exam', 'hpi femal histori',
    'intact cn visual', 'intraven continu infus', 'left gaze prefer', 'left mca syndrom',
    'left no drift', 'left side weak', 'light touch no', 'light touch reflex', 'loc alert respons',
    'loss languag no', 'lower extrem sensat', 'lower facial weak', 'mca territori infarct',
    'medic ed chief', 'memori intact cn', 'motor right upper', 'no drift limb',
    'no dysarthria shoulder', 'not follow command', 'orient time appear', 'pend cta head',
    'present alter mental', 'prn imag head', 'pronat drift strength', 'refer rang cardiac',
    'result pend xr', 'right upper extrem', 'rr bp no', 'rr devic none', 'sc seizur commun',
    'side weak left', 'telemetri laboratori work', 'upper extrem left', 'upper extrem right',
    'urinari cathet less',
]

# Stopwords used during training
_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'youre', 'youve',
    'youll', 'youd', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'shes', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
    'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'go', 'went', 'will', 'can', 'sent',
    'who', 'whom', 'mr', 'dr', 'dear', 'resident', 'physician', 'mrn', 'father', 'mother',
    'provider', 'ehr', 'rn', 'clinician', 'md', 'inpatient', 'patient', 'sister', 'brother',
    'partner', 'ros', 'husband', 'wife', 'spouse', 'person', 'staff', 'name', 'pcp', 'dob',
    'medic', 'doct', 'daughter', 'phd', 'pgy', 'family', 'hcp', 'nurse', 'np', 'partners',
    'patients', 'pta', 'slp', 'therapist', 'son', 'if', 'of', 'ac', 'in', 'out', 'by', 'at',
    'fo', 'nan', 'hea', 'pro', 'as', 'or', 'a', 'an', 'for', 'the', 'with', 'to', 'be',
    'from', 'about', 'should', 'would', 'could', 'same', 'thank', 'please', 'another',
    'either', 'every', 'although', 'but', 'yet', 'this', 'that', 'what', 'there', 'here',
    'these', 'those', 'also', 'as well', 'too', 'because', 'until', 'while', 'when', 'where',
    'why', 'how', 'then', 'throughout', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'once', 'few', 'more', 'ever', 'on', 'off', 'over', 'under', 'again',
    'further', 'and', 'which', 'yes', 'other', 'worst', 'best', 'most', 'some', 'such',
    'only', 'own', 'so', 'than', 'very', 'however', 'even', 'just', 'above', 'below',
    'summary', 'facesheet', 'items', 'code', 'phone', 'visit', 'attend', 'encounter', 'note',
    'admit', 'admission', 'admitted', 'consult', 'file', 'report', 'page', 'pager', 'comment',
    'service', 'relate', 'send', 'edit', 'edited', 'document', 'documentation', 'part', 'use',
    'errors', 'set', 'education', 'study', 'find', 'notify', 'systems', 'review', 'assessment',
    'pertinent', 'contact', 'outside', 'diagnose', 'information', 'index', 'additional',
    'result', 'schedule', 'main', 'former', 'basic', 'comments', 'documented', 'documents',
    'data', 'syring', 'mdd', 'apt', 'resu', 'con', 'dis', 'mch', 'wnl', 'mghe', 'diff',
    'id', 'hs', 'hid', 'post', 'nt', 'tid', 'nad', 'pa', 'na', 'msk', 'cc', 're', 'nih',
    'score', 'cmf', 'vs', 'massachusetts', 'ma', 'address', 'newton', 'highland', 'icu',
    'boston', 'fa', 'street', 'waltham', 'sommerville', 'avenue', 'mghw', 'cambridge',
    'revere', 'charlestown', 'wa', 'barrasso', 'salem', 'lincoln', 'luckhurst', 'st', 'chelsea',
    'mghg', 'nc', 'mccann', 'mgh', 'bwh', 'webster', 'lynn', 'haverhill', 'bi', 'needham',
    'general', 'hospital', 'pgy2', 'department', 'ellison', 'neurology', 'brigham', 'women',
    'logan', 'airport', 'lunder', 'blake', 'winthrop', 'wi', 'place', 'ave', 'date', 'first',
    'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eigth', 'nineth', 'tenth',
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september',
    'october', 'november', 'december', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
    'saturday', 'sunday', 'winter', 'spring', 'summer', 'autumn', 'am', 'pm', 'hour', 'hours',
    'hr', 'hrs', 'day', 'days', 'minutes', 'year', 'years', 'week', 'weeks', 'months', 'month',
    'today', 'currently', 'yesterday', 'yo', 'hourly', 'nightly', 'daily', 'monthly', 'yearly',
    'weekly', 'last', 'now', 'mg', 'ml', 'mmhg', 'g', 'cm', 'ii', 'xii', 'lb', 'xl', 'kg',
    'pmh', 'unit', 'units', 'qh', 'qd', 'per', 'sig', 'bid', 'mm', 'mcg', 'dl', 'mol',
    'mmol', 'neuts', 'oz', 'ng', 'qhs', 'pf', 'wbc', 'hgb', 'rdw', 'rbc', 'cbc', 'plt',
    'ldl', 'gtt', 'bun', 'hdl', 'glu', 'lipid', 'spo', 'lfts', 'hld', 'hbac', 'hct', 'ca',
    'cr', 'creatinine', 'hb', 'hemoglobin', 'phos', 'potassium', 'magnesium', 'phosphate',
    'calcium', 'urea', 'zinc', 'mcv', 'mch', 'mchc', 'sodium', 'chloride', 'ammonia',
    'glucose', 'calc', 'monos', 'eos', 'baso', 'basos', 'neutrophils', 'troponin',
    'cholesterol', 'triglycerides', 'mpv', 'lymph', 'lymphs', 'mono', 'neutrophil', 'chol',
    'nrbc', 'gfr', 'ua', 'cl', 'tp', 'co', 'cre', 'pt', 'inr', 'lact', 'tropt', 'sgpt',
    'sgot', 'ntbnp', 'alkp', 'tbili', 'dbili', 'alb', 'tsh', 'take', 'tab', 'tablet',
    'tablets', 'vitals', 'lab', 'labs', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k',
    'm', 'n', 'o', 'p', 'q', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
}

# NIHSS structured form substitutions: convert numeric NIHSS codes to descriptive phrases
_N2W_PATTERNS = [
    # LOC alert
    (r'alert\s?0\s?\(0-3\)', 'alert keenly responsive'),
    (r'alert\s?1\s?\(0-3\)', 'arouses to minor stimulation'),
    (r'alert\s?2\s?\(0-3\)', 'movements to pain'),
    (r'alert\s?3\s?\(0-3\)', 'unresponsive'),
    (r'1a- 0', 'alert keenly responsive'),
    (r'1a- 1', 'arouses to minor stimulation'),
    (r'1a- 2', 'movements to pain'),
    (r'1a- 3', 'unresponsive'),
    # LOC questions
    (r'question\s?0\s?\(0-2\)', 'answers both questions correctly'),
    (r'question\s?1\s?\(0-2\)', 'dysarthric'),
    (r'question\s?2\s?\(0-2\)', 'aphasic'),
    (r'1b- 0', 'answers both questions correctly'),
    (r'1b- 1', 'dysarthric'),
    (r'1b- 2', 'aphasic'),
    # LOC commands
    (r'commands\s?0\s?\(0-2\)', 'performs both tasks correctly'),
    (r'commands\s?1\s?\(0-2\)', 'performs one task'),
    (r'commands\s?2\s?\(0-2\)', 'not perform tasks'),
    (r'1c- 0', 'performs both tasks correctly'),
    (r'1c- 1', 'performs one task'),
    (r'1c- 2', 'not perform tasks'),
    # Gaze
    (r'gaze\s?0\s?\(0-2\)', 'gaze normal'),
    (r'gaze\s?1\s?\(0-2\)', 'partial gaze palsy'),
    (r'gaze\s?2\s?\(0-2\)', 'forced gaze palsy'),
    (r'gaze 2- 0', 'gaze normal'),
    (r'gaze 2- 1', 'partial gaze palsy'),
    (r'gaze 2- 2', 'forced gaze palsy'),
    # Visual field
    (r'visual field\s?0\s?\(0-3\)', 'no visual loss'),
    (r'visual field\s?1\s?\(0-3\)', 'partial hemianopia'),
    (r'visual field\s?2\s?\(0-3\)', 'complete hemianopia'),
    (r'visual field\s?3\s?\(0-3\)', 'bilateral hemianopia'),
    # Facial palsy
    (r'facial palsy\s?0\s?\(0-3\)', 'normal symmetrical movements'),
    (r'facial palsy\s?1\s?\(0-3\)', 'minor paralysis'),
    (r'facial palsy\s?2\s?\(0-3\)', 'partial paralysis'),
    (r'facial palsy\s?3\s?\(0-3\)', 'complete paralysis'),
    # Motor arms/legs (abbreviated)
    (r'motor left arm\s?0\s?\(0-4\)', 'mlarm no drift'),
    (r'motor left arm\s?1\s?\(0-4\)', 'mlarm drift'),
    (r'motor left arm\s?[234]\s?\(0-4\)', 'mlarm no effort'),
    (r'motor right arm\s?0\s?\(0-4\)', 'mrarm no drift'),
    (r'motor right arm\s?1\s?\(0-4\)', 'mrarm drift'),
    (r'motor right arm\s?[234]\s?\(0-4\)', 'mrarm no effort'),
    (r'motor left leg\s?0\s?\(0-4\)', 'mlleg no drift'),
    (r'motor left leg\s?1\s?\(0-4\)', 'mlleg drift'),
    (r'motor left leg\s?[234]\s?\(0-4\)', 'mlleg no effort'),
    (r'motor right leg\s?0\s?\(0-4\)', 'mrleg no drift'),
    (r'motor right leg\s?1\s?\(0-4\)', 'mrleg drift'),
    (r'motor right leg\s?[234]\s?\(0-4\)', 'mrleg no effort'),
    # Ataxia
    (r'ataxia\s?0\s?\(0-2\)', 'no ataxia'),
    (r'ataxia\s?1\s?\(0-2\)', 'ataxia in one limb'),
    (r'ataxia\s?2\s?\(0-2\)', 'ataxia in two limbs'),
    # Sensory
    (r'sensory\s?0\s?\(0-2\)', 'no sensory loss'),
    (r'sensory\s?1\s?\(0-2\)', 'mild moderate loss'),
    (r'sensory\s?2\s?\(0-2\)', 'no response'),
    # Language
    (r'language\s?0\s?\(0-3\)', 'no aphasia'),
    (r'language\s?1\s?\(0-3\)', 'mild moderate aphasia'),
    (r'language\s?2\s?\(0-3\)', 'severe aphasia'),
    (r'language\s?3\s?\(0-3\)', 'unresponsive'),
    # Dysarthria
    (r'dysarthria\s?0\s?\(0-2\)', 'normal speech'),
    (r'dysarthria\s?1\s?\(0-2\)', 'mild moderate dysarthria'),
    (r'dysarthria\s?2\s?\(0-2\)', 'severe dysarthria'),
    # Inattention/Extinction
    (r'inattention\s?0\s?\(0-2\)', 'no abnormality'),
    (r'inattention\s?1\s?\(0-2\)', 'visual tactile auditory spatial personal inattention'),
    (r'inattention\s?2\s?\(0-2\)', 'profound hemi inattention'),
    (r'extinction\s?0\s?\(0-2\)', 'no abnormality'),
    (r'extinction\s?1\s?\(0-2\)', 'visual tactile auditory spatial personal inattention'),
    (r'extinction\s?2\s?\(0-2\)', 'profound hemi inattention'),
    (r'extinction\/?(inattention)? 0', 'no abnormality'),
    (r'extinction\/?(inattention)? 1', 'visual tactile auditory spatial personal inattention'),
    (r'extinction\/?(inattention)? 2', 'profound hemi inattention'),
]
_N2W_COMPILED = [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in _N2W_PATTERNS]

# Abbreviation expansion/contraction pairs (applied in order)
_ABBREV_EXPANSIONS = [
    ('w/out', 'without'), ('w out', 'without'), ('w/ out', 'without'),
    ("n't", ' not'), ('neither', 'not'), (' nor ', ' not '),
    (' w/', ' with '), (' r/o ', ' rule out '), (' h/o ', ' history of '),
    (' a-fib ', ' afib '), (' a fib ', ' afib '),
    (' l ', ' left '), (' r ', ' right '), (' hx ', ' history '),
    (' dispo ', ' disposition '), (' temp ', ' temperature '),
    (' gi ', ' gastrointestinal '), (' gu ', ' genitourinary '),
    (' resp ', ' respiratory '), (' eval ', ' evaluation '),
    (' meds ', ' medications '), ('medicines', 'medications'), ('medicine', 'medication'),
    (' cv ', ' cardiovascular '), (' htn ', ' hypertension '),
    (' iv ', ' intravenous '), (' gen ', ' general '), (' dx ', ' diagnosis '),
    (' pulm ', ' pulmonary '), (' ref ', ' refer '), (' tx ', ' treatment '),
    (' dm ', ' diabetes '), (' abd ', ' abdominal '), ('abdomen', 'abdominal'),
    (' ext ', ' extremities '), (' extr ', ' extremities '),
    (' ecg ', ' electrocardiogram '), (' ekg ', ' electrocardiogram '),
    (' echo ', ' echocardiogram '), (' etoh ', ' alcohol '), (' wt ', ' weight '),
    (' ng tube ', ' nasogastric tube '), (' ngtube ', ' nasogastric tube '),
    (' max ', ' maximum '), (' min ', ' minimum '), (' neg ', ' negative '),
    (' ms ', ' mental status '), (' onc ', ' oncology '),
    (' ot ', ' occupational therapy '),
    (' lle ', ' left lower extremity '), (' lue ', ' left upper extremity '),
    (' rue ', ' right upper extremity '), (' rle ', ' right lower extremity '),
    (' le ', ' lower extremity '), (' ue ', ' upper extremity '),
    (' lsw ', ' left sided weakness '), (' rsw ', ' right sided weakness '),
    (' lkw ', ' last known well '), (' ams ', ' altered mental status '),
    (' ddx ', ' differential diagnosis '),
    (' nad ', ' no abnormality detected no apparent distress no appreciable disease '),
    (' regular rate and rhythm ', ' rrr '), (' regular rate and regular rhythm ', ' rrr '),
    (' regular rate regular rhythm ', ' rrr '),
    ('normal saline', 'ns'), ('blood pressure', 'bp'), ('heart rate', 'hr'),
    ('respiratory rate', 'rr'), ('atrial fibrillation', 'afib'),
    ('emergency department', 'ed'), ('nothing by mouth', 'npo'),
    ('outside hospital', 'osh'), ('middle cerebral artery', 'mca'),
    ('cranial nerves', 'cn'), ('cranial nerve', 'cn'),
    ('computed tomography', 'ct'), ('ct angiography', 'cta'), ('ct angio', 'cta'),
    (' angio ', ' angiography '), ('magnetic resonance angiography', 'mra'),
    ('deep vein thrombosis', 'dvt'), ('erythrocyte sedimentation rate', 'esr'),
    ('history of present illness', 'hpi'), ('internal carotid artery', 'ica'),
    ('partial thromboplastin time', 'ptt'), ('transthoracic echocardiogram', 'tte'),
    ('tissue plasminogen activator', 'tpa'), ('room air', 'ra'),
    ('review of systems', 'ros'), ('speech language pathologist', 'slp'),
    ('primary care physician', 'pcp'), ('chest radiography', 'cxr'),
    ('diabetes mellitus', 'diabetes'), ('past medical history', 'pmhx'),
    ('physical therapist assistant', 'pta'), ('range of motion', 'rom'),
    ('shortness of breath', 'sob'), ('shortness breath', 'sob'),
    ('transient ischaemic attack', 'tia'), ('transient ischemic attack', 'tia'),
    ('congestive heart failure', 'chf'), ('intra arterial therapy', 'iat'),
    ('intraarterial therapy', 'iat'), ('subarachnoid hemorrhage ', 'sah'),
    ('head of bed', 'hob'), ('out of bed', 'oob'),
    ('emergency medical services', 'ems'), ('level of consciousness', 'loc'),
    ('level consciousness', 'loc'), ('social history', 'sh'),
    ('murmurs rubs and gallops', 'mgr'), ('murmurs rubs or gallops', 'mgr'),
    ('murmurs rubs gallops', 'mgr'), ('murmurs gallops or rubs', 'mgr'),
    ('murmurs gallops rubs', 'mgr'), (' mrg ', ' mgr '),
    ('medical decision making', 'mdm'), ('mean corpuscular volume', 'mcv'),
    ('familial hypercholesterolemia', 'fh'), (' xray ', ' xr '), (' x ray ', ' xr '),
    ('by mouth', 'po'), ('cerebral vascular accident', 'cva'),
    ('cerebralvascular accident', 'cva'), ('coronary artery disease', 'cad'),
    ('ear nose and throat', 'ent'), ('ear nose throat', 'ent'),
    ('head eyes ent', 'heent'),
    (' mlarm ', ' motor left arm '), (' mrarm ', ' motor right arm '),
    (' mlleg ', ' motor left leg '), (' mrleg ', ' motor right leg '),
]

_NIHSS_REMOVE = [
    'nihss stroke scale', 'nihss stroke score', 'stroke scale', 'stroke score',
    'nihss scale', 'nihss score', 'nihss total score', 'nih telestroke scale total',
    'nihss', ' vs ', ' ec tablets', ' ec tablet', 'vital signs', 'vital sign',
]


def _preprocess_text(text: str, stemmer: PorterStemmer) -> str:
    """Apply the full NIHSS preprocessing pipeline to a single note."""
    text = text.lower()

    # Abbreviation expansions
    for src, tgt in _ABBREV_EXPANSIONS:
        text = text.replace(src, tgt)

    # NIHSS number-to-words
    for pattern, replacement in _N2W_COMPILED:
        text = pattern.sub(replacement, text)

    # Remove numbers
    text = re.sub(r'\d', '', text)

    # Remove special characters (keep letters, digits, space, period)
    text = text.replace('/l', ' ')
    text = re.sub(r'[^a-zA-Z0-9 \n.]', ' ', text)
    text = text.replace('.', ' ')
    text = re.sub(r'\s+', ' ', text)

    # Remove duplicate consecutive words
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    # Remove NIHSS score references
    for expr in _NIHSS_REMOVE:
        text = text.replace(expr, ' ')

    # Stopword removal (word_tokenize handles punctuation tokens too)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) >= 1]
    text = ' '.join(tokens)

    # Remove duplicated spaces and words again
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    # Porter stemming
    words = text.split()
    stemmed = [stemmer.stem(w) for w in words]
    text = ' '.join(stemmed).strip()

    # Final dedup
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    return text


def _extract_features(processed_text: str) -> dict:
    """Binary presence of each NIHSS feature n-gram in preprocessed text."""
    result = {}
    for feat in NIHSS_FEATURES:
        result[feat] = 1 if re.search(r'(?<![a-z])' + re.escape(feat) + r'(?![a-z])', processed_text) else 0
    return result


class NIHSSModel(_BaseModel):
    path = importlib.resources.files("prophet.models.nihss").joinpath("config.yaml")
    if path.exists():
        DEFAULT_CONFIG_PATH = str(path)
    else:
        raise FileNotFoundError(f"Configuration file not found: {path}")

    def __init__(self, config_path=None, external_ray=False):
        if config_path is None:
            config_path = self.DEFAULT_CONFIG_PATH
        super().__init__(config_path)
        self.external_ray = external_ray

        for resource in ('tokenizers/punkt', 'tokenizers/punkt_tab'):
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.download('punkt_tab')
                break

    def load_model(self, model_path: str):
        logger.info(f"Loading model from {model_path}")
        path = importlib.resources.files("prophet.models.nihss").joinpath(model_path)
        return joblib.load(str(path))

    def run(self, data: Dict[str, pl.DataFrame], show_progress=False, return_features=False, force_casting=False) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
        feat = self.preprocess(data, show_progress, force_casting)
        pred = self.predict(feat)
        if return_features:
            return feat, pred
        return pred

    def preprocess(self, data: Dict[str, pl.DataFrame], show_progress=False, force_casting=False) -> pl.DataFrame:
        logger.info(f"Preprocessing started at {datetime.now()}")
        data = super().preprocess(data, show_progress, force_casting)
        # note_idx is the stable per-note key (row position in the input note file).
        # date is passed through so annotations can be joined back on id+date+note_idx.
        data['note'] = data['note'].with_row_index('note_idx')

        feat = data['note'].select(['note_idx', 'id', 'date'])
        if len(feat) == 0:
            raise ValueError("No notes found in the provided data.")
        logger.info(f"Generating features for n={len(feat)} notes")

        note_feat = self._preprocess_notes(data['note'], feat, show_progress)
        logger.info(f"Preprocessing finished at {datetime.now()}")
        return note_feat

    def _preprocess_notes(self, note_df: pl.DataFrame, feat: pl.DataFrame, show_progress=False) -> pl.DataFrame:
        ray_was_initialized = ray.is_initialized()
        if not ray_was_initialized and not self.external_ray:
            ray.init(
                object_store_memory=int(0.2 * psutil.virtual_memory().total),
                _memory=int(0.3 * psutil.virtual_memory().total),
                _redis_max_memory=int(0.05 * psutil.virtual_memory().total),
            )
            should_shutdown = True
        else:
            should_shutdown = False

        try:
            batch_size = 500
            num_cpus = int(ray.available_resources().get("CPU", 1))
            total_notes = len(note_df)
            temp_dir = Path(tempfile.mkdtemp(prefix="nihss_processing_"))

            if show_progress:
                from tqdm import tqdm
                progress_bar = tqdm(total=total_notes, desc='Processing NIHSS notes')
            else:
                progress_bar = None

            @ray.remote(num_cpus=1)
            class NIHSSProcessor:
                def __init__(self):
                    import nltk
                    from nltk.stem import PorterStemmer
                    self.stemmer = PorterStemmer()

                def process_batch(self, batch_data, batch_id, temp_dir_str):
                    batch_results = []
                    try:
                        for item in batch_data:
                            text = item['note'] or ''
                            processed = _preprocess_text(text, self.stemmer)
                            row = {'index': item['note_idx']}
                            row.update(_extract_features(processed))
                            batch_results.append(row)

                        batch_df = pl.DataFrame(batch_results)
                        output_path = Path(temp_dir_str) / f"batch_{batch_id}.parquet"
                        batch_df.write_parquet(output_path)
                        del batch_results, batch_df
                        gc.collect()
                        return len(batch_data), str(output_path)
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_id}: {e}")
                        raise

            max_concurrent_actors = max(1, num_cpus - 2)
            processors = [NIHSSProcessor.remote() for _ in range(max_concurrent_actors)]
            notes_dicts = note_df.select(['note_idx', 'note']).to_dicts()
            batches = [notes_dicts[i:i + batch_size] for i in range(0, len(notes_dicts), batch_size)]

            logger.info(f"Processing {total_notes} notes in {len(batches)} batches using {max_concurrent_actors} actors")

            futures = []
            batch_counter = 0
            completed_files = []

            for i, batch in enumerate(batches):
                processor = processors[i % len(processors)]
                future = processor.process_batch.remote(batch, batch_counter, str(temp_dir))
                futures.append(future)
                batch_counter += 1

                if len(futures) >= max_concurrent_actors * 2:
                    ready, futures = ray.wait(futures, num_returns=1, timeout=None)
                    for ref in ready:
                        try:
                            count, path = ray.get(ref)
                            completed_files.append(path)
                            if progress_bar:
                                progress_bar.update(count)
                        except Exception as e:
                            logger.error(f"Error getting result: {e}")

            while futures:
                ready, futures = ray.wait(futures, num_returns=len(futures), timeout=60)
                for ref in ready:
                    try:
                        count, path = ray.get(ref, timeout=30)
                        completed_files.append(path)
                        if progress_bar:
                            progress_bar.update(count)
                    except ray.exceptions.GetTimeoutError:
                        logger.warning("Task timed out during final processing")
                    except Exception as e:
                        logger.error(f"Error in final processing: {e}")

            if progress_bar:
                progress_bar.close()

            for processor in processors:
                ray.kill(processor)
            gc.collect()

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise
        finally:
            if should_shutdown:
                ray.shutdown()
            gc.collect()

        try:
            parquet_files = list(temp_dir.glob("batch_*.parquet"))
            if not parquet_files:
                raise ValueError("No batch files were created")

            note_feat = pl.read_parquet(parquet_files)
            note_feat = note_feat.rename({'index': 'note_idx'})
            note_feat = note_feat.join(note_df.select(['note_idx', 'id', 'date']), on='note_idx', how='left')
            note_feat = feat.join(note_feat, on=['note_idx', 'id', 'date'], how='left', validate='1:1').fill_null(0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return note_feat

    def predict(self, feat: pl.DataFrame) -> pl.DataFrame:
        _start = time.time()
        logger.info(f"Prediction started at {datetime.now().strftime('%H:%M:%S')}")

        X = feat.select(NIHSS_FEATURES).to_numpy()
        scores = self.model.predict(X).clip(0, 42)

        pred = feat.select(['note_idx', 'id', 'date']).with_columns(
            pl.Series('nihss_score', scores)
        )

        logger.info(f"Prediction finished in {time.time() - _start:.2f}s")
        return pred
