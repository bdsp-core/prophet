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

# Stopwords used during training (identical to NIHSS)
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
    'these', 'those', 'also', 'too', 'because', 'until', 'while', 'when', 'where', 'why',
    'how', 'then', 'throughout', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'once', 'few', 'more', 'ever', 'on', 'off', 'over', 'under', 'again', 'further',
    'and', 'which', 'yes', 'other', 'worst', 'best', 'most', 'some', 'such', 'only', 'own',
    'so', 'than', 'very', 'however', 'even', 'just', 'above', 'below',
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
    'revere', 'charlestown', 'wa', 'barrasso', 'salem', 'lincoln', 'luckhurst', 'st',
    'chelsea', 'mghg', 'nc', 'mccann', 'mgh', 'bwh', 'webster', 'lynn', 'haverhill', 'bi',
    'needham', 'general', 'hospital', 'pgy2', 'department', 'ellison', 'neurology', 'brigham',
    'women', 'logan', 'airport', 'lunder', 'blake', 'winthrop', 'wi', 'place', 'ave',
    'date', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eigth',
    'nineth', 'tenth', 'january', 'february', 'march', 'april', 'may', 'june', 'july',
    'august', 'september', 'october', 'november', 'december', 'monday', 'tuesday',
    'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'winter', 'spring', 'summer',
    'autumn', 'am', 'pm', 'hour', 'hours', 'hr', 'hrs', 'day', 'days', 'minutes', 'year',
    'years', 'week', 'weeks', 'months', 'month', 'today', 'currently', 'yesterday', 'yo',
    'hourly', 'nightly', 'daily', 'monthly', 'yearly', 'weekly', 'last', 'now',
    'mg', 'ml', 'mmhg', 'g', 'cm', 'ii', 'xii', 'lb', 'xl', 'kg', 'pmh', 'unit', 'units',
    'qh', 'qd', 'per', 'sig', 'bid', 'mm', 'mcg', 'dl', 'mol', 'mmol', 'neuts', 'oz', 'ng',
    'qhs', 'pf', 'wbc', 'hgb', 'rdw', 'rbc', 'cbc', 'plt', 'ldl', 'gtt', 'bun', 'hdl',
    'glu', 'lipid', 'spo', 'lfts', 'hld', 'hbac', 'hct', 'ca', 'cr', 'creatinine', 'hb',
    'hemoglobin', 'phos', 'potassium', 'magnesium', 'phosphate', 'calcium', 'urea', 'zinc',
    'mcv', 'mch', 'mchc', 'sodium', 'chloride', 'ammonia', 'glucose', 'calc', 'monos', 'eos',
    'baso', 'basos', 'neutrophils', 'troponin', 'cholesterol', 'triglycerides', 'mpv', 'lymph',
    'lymphs', 'mono', 'neutrophil', 'chol', 'nrbc', 'gfr', 'ua', 'cl', 'tp', 'co', 'cre',
    'pt', 'inr', 'lact', 'tropt', 'sgpt', 'sgot', 'ntbnp', 'alkp', 'tbili', 'dbili', 'alb',
    'tsh', 'take', 'tab', 'tablet', 'tablets', 'vitals', 'lab', 'labs',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'o', 'p', 'q', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z',
}

# NIHSS structured form substitutions (full severity levels, matching the original
# NIHSS_numbers2words.py used for the mRS paper's preprocessing)
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
    (r'0 1\s?b', 'alert keenly responsive'),
    (r'1 1\s?b', 'arouses to minor stimulation'),
    (r'2 1\s?b', 'movements to pain'),
    (r'3 1\s?b', 'unresponsive'),
    # LOC questions
    (r'question\s?0\s?\(0-2\)', 'answers both questions correctly'),
    (r'question\s?1\s?\(0-2\)', 'dysarthric'),
    (r'question\s?2\s?\(0-2\)', 'aphasic'),
    (r'1b- 0', 'answers both questions correctly'),
    (r'1b- 1', 'dysarthric'),
    (r'1b- 2', 'aphasic'),
    (r'0 1\s?c', 'answers both questions correctly'),
    (r'1 1\s?c', 'dysarthric'),
    (r'2 1\s?c', 'aphasic'),
    # LOC commands
    (r'commands\s?0\s?\(0-2\)', 'performs both tasks correctly'),
    (r'commands\s?1\s?\(0-2\)', 'performs one task'),
    (r'commands\s?2\s?\(0-2\)', 'not perform tasks'),
    (r'1c- 0', 'performs both tasks correctly'),
    (r'1c- 1', 'performs one task'),
    (r'1c- 2', 'not perform tasks'),
    (r'0 2 (best)?\s?gaze', 'performs both tasks correctly'),
    (r'1 2 (best)?\s?gaze', 'performs one task'),
    (r'2 2 (best)?\s?gaze', 'not perform tasks'),
    # Horizontal gaze
    (r'gaze\s?0\s?\(0-2\)', 'gaze normal'),
    (r'gaze\s?1\s?\(0-2\)', 'partial gaze palsy'),
    (r'gaze\s?2\s?\(0-2\)', 'forced gaze palsy'),
    (r'gaze 2- 0', 'gaze normal'),
    (r'gaze 2- 1', 'partial gaze palsy'),
    (r'gaze 2- 2', 'forced gaze palsy'),
    (r'0 3 visual fields?', 'gaze normal'),
    (r'1 3 visual fields?', 'partial gaze palsy'),
    (r'2 3 visual fields?', 'forced gaze palsy'),
    # Visual field
    (r'visual field\s?0\s?\(0-3\)', 'no visual loss'),
    (r'visual field\s?1\s?\(0-3\)', 'partial hemianopia'),
    (r'visual field\s?2\s?\(0-3\)', 'complete hemianopia'),
    (r'visual field\s?3\s?\(0-3\)', 'bilateral hemianopia'),
    (r'visual (field)?\s?3- 0', 'no visual loss'),
    (r'visual (field)?\s?3- 1', 'partial hemianopia'),
    (r'visual (field)?\s?3- 2', 'complete hemianopia'),
    (r'visual (field)?\s?3- 3', 'bilateral hemianopia'),
    (r'0 4 facial palsy', 'no visual loss'),
    (r'1 4 facial palsy', 'partial hemianopia'),
    (r'2 4 facial palsy', 'complete hemianopia'),
    (r'3 4 facial palsy', 'bilateral hemianopia'),
    # Facial palsy
    (r'facial palsy\s?0\s?\(0-3\)', 'normal symmetrical movements'),
    (r'facial palsy\s?1\s?\(0-3\)', 'minor paralysis'),
    (r'facial palsy\s?2\s?\(0-3\)', 'partial paralysis'),
    (r'facial palsy\s?3\s?\(0-3\)', 'complete paralysis'),
    (r'facial palsy 4- 0', 'normal symmetrical movements'),
    (r'facial palsy 4- 1', 'minor paralysis'),
    (r'facial palsy 4- 2', 'partial paralysis'),
    (r'facial palsy 4- 3', 'complete paralysis'),
    (r'0 5\s?a', 'normal symmetrical movements'),
    (r'1 5\s?a', 'minor paralysis'),
    (r'2 5\s?a', 'partial paralysis'),
    (r'3 5\s?a', 'complete paralysis'),
    # Motor left arm
    (r'motor left arm\s?0\s?\(0-4\)', 'mlarm no drift'),
    (r'motor left arm\s?1\s?\(0-4\)', 'mlarm drift'),
    (r'motor left arm\s?2\s?\(0-4\)', 'mlarm drift hits bed'),
    (r'motor left arm\s?3\s?\(0-4\)', 'mlarm no effort'),
    (r'motor left arm\s?4\s?\(0-4\)', 'mlarm no movement'),
    (r'motor left arm 5a- 0', 'mlarm no drift'),
    (r'motor left arm 5a- 1', 'mlarm drift'),
    (r'motor left arm 5a- 2', 'mlarm drift hits bed'),
    (r'motor left arm 5a- 3', 'mlarm no effort'),
    (r'motor left arm 5a- 4', 'mlarm no movement'),
    (r'0 5\s?b', 'mlarm no drift'),
    (r'1 5\s?b', 'mlarm drift'),
    (r'2 5\s?b', 'mlarm drift hits bed'),
    (r'3 5\s?b', 'mlarm no effort'),
    (r'4 5\s?b', 'mlarm no movement'),
    # Motor right arm
    (r'motor right arm\s?0\s?\(0-4\)', 'mrarm no drift'),
    (r'motor right arm\s?1\s?\(0-4\)', 'mrarm drift'),
    (r'motor right arm\s?2\s?\(0-4\)', 'mrarm drift hits bed'),
    (r'motor right arm\s?3\s?\(0-4\)', 'mrarm no effort'),
    (r'motor right arm\s?4\s?\(0-4\)', 'mrarm no movement'),
    (r'motor right arm 5b- 0', 'mrarm no drift'),
    (r'motor right arm 5b- 1', 'mrarm drift'),
    (r'motor right arm 5b- 2', 'mrarm drift hits bed'),
    (r'motor right arm 5b- 3', 'mrarm no effort'),
    (r'motor right arm 5b- 4', 'mrarm no movement'),
    (r'0 6\s?a', 'mrarm no drift'),
    (r'1 6\s?a', 'mrarm drift'),
    (r'2 6\s?a', 'mrarm drift hits bed'),
    (r'3 6\s?a', 'mrarm no effort'),
    (r'4 6\s?a', 'mrarm no movement'),
    # Motor left leg
    (r'motor left leg\s?0\s?\(0-4\)', 'mlleg no drift'),
    (r'motor left leg\s?1\s?\(0-4\)', 'mlleg drift'),
    (r'motor left leg\s?2\s?\(0-4\)', 'mlleg drift hits bed'),
    (r'motor left leg\s?3\s?\(0-4\)', 'mlleg no effort'),
    (r'motor left leg\s?4\s?\(0-4\)', 'mlleg no movement'),
    (r'motor left leg 6a- 0', 'mlleg no drift'),
    (r'motor left leg 6a- 1', 'mlleg drift'),
    (r'motor left leg 6a- 2', 'mlleg drift hits bed'),
    (r'motor left leg 6a- 3', 'mlleg no effort'),
    (r'motor left leg 6a- 4', 'mlleg no movement'),
    (r'0 6\s?b', 'mlleg no drift'),
    (r'1 6\s?b', 'mlleg drift'),
    (r'2 6\s?b', 'mlleg drift hits bed'),
    (r'3 6\s?b', 'mlleg no effort'),
    (r'4 6\s?b', 'mlleg no movement'),
    # Motor right leg
    (r'motor right leg\s?0\s?\(0-4\)', 'mrleg no drift'),
    (r'motor right leg\s?1\s?\(0-4\)', 'mrleg drift'),
    (r'motor right leg\s?2\s?\(0-4\)', 'mrleg drift hits bed'),
    (r'motor right leg\s?3\s?\(0-4\)', 'mrleg no effort'),
    (r'motor right leg\s?4\s?\(0-4\)', 'mrleg no movement'),
    (r'motor right leg 6b- 0', 'mrleg no drift'),
    (r'motor right leg 6b- 1', 'mrleg drift'),
    (r'motor right leg 6b- 2', 'mrleg drift hits bed'),
    (r'motor right leg 6b- 3', 'mrleg no effort'),
    (r'motor right leg 6b- 4', 'mrleg no movement'),
    (r'0 7 limb ataxia', 'mrleg no drift'),
    (r'1 7 limb ataxia', 'mrleg drift'),
    (r'2 7 limb ataxia', 'mrleg drift hits bed'),
    (r'3 7 limb ataxia', 'mrleg no effort'),
    (r'4 7 limb ataxia', 'mrleg no movement'),
    # Ataxia
    (r'ataxia\s?0\s?\(0-2\)', 'no ataxia'),
    (r'ataxia\s?1\s?\(0-2\)', 'ataxia in one limb'),
    (r'ataxia\s?2\s?\(0-2\)', 'ataxia in two limbs'),
    (r'ataxia 7- 0', 'no ataxia'),
    (r'ataxia 7- 1', 'ataxia in one limb'),
    (r'ataxia 7- 2', 'ataxia in two limbs'),
    (r'0 8 sensory', 'no ataxia'),
    (r'1 8 sensory', 'ataxia in one limb'),
    (r'2 sensory', 'ataxia in two limbs'),
    # Sensory
    (r'sensory\s?0\s?\(0-2\)', 'no sensory loss'),
    (r'sensory\s?1\s?\(0-2\)', 'mild moderate loss'),
    (r'sensory\s?2\s?\(0-2\)', 'no response'),
    (r'sensory 8- 0', 'no sensory loss'),
    (r'sensory 8- 1', 'mild moderate loss'),
    (r'sensory 8- 2', 'no response'),
    (r'0 9 (best)?\s?language', 'no sensory loss'),
    (r'1 9 (best)?\s?language', 'mild moderate loss'),
    (r'2 9 (best)?\s?language', 'no response'),
    # Language
    (r'language\s?0\s?\(0-3\)', 'no aphasia'),
    (r'language\s?1\s?\(0-3\)', 'mild moderate aphasia'),
    (r'language\s?2\s?\(0-3\)', 'severe aphasia'),
    (r'language\s?3\s?\(0-3\)', 'unresponsive'),
    (r'language 9- 0', 'no aphasia'),
    (r'language 9- 1', 'mild moderate aphasia'),
    (r'language 9- 2', 'severe aphasia'),
    (r'language 9- 3', 'unresponsive'),
    (r'0 10 dysarthria', 'no aphasia'),
    (r'1 10 dysarthria', 'mild moderate aphasia'),
    (r'2 10 dysarthria', 'severe aphasia'),
    (r'3 10 dysarthria', 'unresponsive'),
    # Dysarthria
    (r'dysarthria\s?0\s?\(0-2\)', 'normal speech'),
    (r'dysarthria\s?1\s?\(0-2\)', 'mild moderate dysarthria'),
    (r'dysarthria\s?2\s?\(0-2\)', 'severe dysarthria'),
    (r'dysarthria 10- 0', 'normal speech'),
    (r'dysarthria 10- 1', 'mild moderate dysarthria'),
    (r'dysarthria 10- 2', 'severe dysarthria'),
    (r'0 11 extinction', 'normal speech'),
    (r'1 11 extinction', 'mild moderate dysarthria'),
    (r'2 11 extinction', 'severe dysarthria'),
    # Extinction and inattention
    (r'inattention\s?0\s?\(0-2\)', 'no abnormality'),
    (r'inattention\s?1\s?\(0-2\)', 'visual tactile auditory spatial personal inattention'),
    (r'inattention\s?2\s?\(0-2\)', 'profound hemi inattention'),
    (r'inattention 11- 0', 'no abnormality'),
    (r'inattention 11- 1', 'visual tactile auditory spatial personal inattention'),
    (r'inattention 11- 2', 'profound hemi inattention'),
    (r'extinction\s?0\s?\(0-2\)', 'no abnormality'),
    (r'extinction\s?1\s?\(0-2\)', 'visual tactile auditory spatial personal inattention'),
    (r'extinction\s?2\s?\(0-2\)', 'profound hemi inattention'),
    (r'extinction 11- 0', 'no abnormality'),
    (r'extinction 11- 1', 'visual tactile auditory spatial personal inattention'),
    (r'extinction 11- 2', 'profound hemi inattention'),
    (r'extinction\/?(inattention)? 0', 'no abnormality'),
    (r'extinction\/?(inattention)? 1', 'visual tactile auditory spatial personal inattention'),
    (r'extinction\/?(inattention)? 2', 'profound hemi inattention'),
]
_N2W_COMPILED = [(re.compile(p), r) for p, r in _N2W_PATTERNS]

# Negation handling + first left/right expansion — applied *before* N2W and digit
# stripping, matching the original preprocessing() order.
_NEGATION_EXPANSIONS = [
    ('w/out', 'without'), ('w out', 'without'), ('w/ out', 'without'),
    ("n't", ' not'), ('neither', 'not'), (' nor ', ' not '),
    (' w/', ' with '), (' r/o ', ' rule out '), (' h/o ', ' history of '),
    (' a-fib ', ' afib '), (' a fib ', ' afib '),
    (' l ', ' left '), (' r ', ' right '),
]

# Abbreviation expansion/contraction pairs — applied *after* N2W and digit/special-char
# stripping, matching the original preprocessing() order (order matters: e.g.
# 'blood pressure'->'bp' must run before 'systolic bp'->'sbp' so that "systolic
# blood pressure" resolves correctly).
_ABBREV_EXPANSIONS = [
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
    (' rsw ', ' right sided weakness '),
    (' lkw ', ' last known well '), (' ams ', ' altered mental status '),
    (' ddx ', ' differential diagnosis '),
    ('cva tenderness', 'costovertebral angle tenderness'),
    (' nad ', ' no abnormality detected no apparent distress no appreciable disease '),
    (' regular rate and rhythm ', ' rrr '), (' regular rate and regular rhythm ', ' rrr '),
    (' regular rate regular rhythm ', ' rrr '),
    ('normal saline', 'ns'), ('blood pressure', 'bp'),
    ('systolic bp', 'sbp'), ('diastolic bp', 'dbp'),
    ('heart rate', 'hr'),
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
    ('ear nose and throat', 'ent'), ('ear nose throat', 'ent'), ('head eyes ent', 'heent'),
    (' mlarm ', ' motor left arm '), (' mrarm ', ' motor right arm '),
    (' mlleg ', ' motor left leg '), (' mrleg ', ' motor right leg '),
]

# mRS-specific leak-prevention terms: strips literal mentions of the outcome label
# itself (e.g. "Modified Rankin Scale: 3") so the model can't read the score off
# the note text. Matches the original mRS preprocessing() removal list exactly
# (this previously reused the unrelated NIHSS removal list by mistake).
_MRS_REMOVE = [
    'modified rankin', ' mrs ', ' rankin ', 'scale', 'score', ' vs ',
    ' ec tablets', ' ec tablet', 'vital signs', 'vital sign',
]


def _preprocess_text(text: str, stemmer: PorterStemmer) -> str:
    """Apply the full preprocessing pipeline to a single note (matches the
    original mRS paper's Preprocessing_function.py / NIHSS_numbers2words.py)."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)

    # Negation + first left/right pass, before N2W (matches original order)
    for src, tgt in _NEGATION_EXPANSIONS:
        text = text.replace(src, tgt)
    text = text.replace(',', ' ')
    text = re.sub(r'\s+', ' ', text)

    # NIHSS structured-field numbers-to-words expansion
    for pattern, replacement in _N2W_COMPILED:
        text = pattern.sub(replacement, text)

    text = re.sub(r'\d', '', text)
    text = text.replace('/l', ' ')
    text = re.sub(r'[^a-zA-Z0-9 \n.]', ' ', text)
    text = text.replace('.', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    # Abbreviation expansion/contraction, after N2W/digit removal (matches original order)
    for src, tgt in _ABBREV_EXPANSIONS:
        text = text.replace(src, tgt)

    for expr in _MRS_REMOVE:
        text = text.replace(expr, ' ')

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) >= 1]
    text = ' '.join(tokens)

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    words = text.split()
    stemmed = [stemmer.stem(w) for w in words]
    text = ' '.join(stemmed).strip()

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    return text


# --- Rule-based score extraction (adapted from the original researcher code's
# 3_Extraction_mrs.py) -------------------------------------------------------
#
# The original pipeline never lets the ML model predict a follow-up mRS when a
# clinician has explicitly documented one in the note: it regex-extracts the
# literal value (binarized: good/poor) and substitutes it for the model's
# prediction, forcing the reported probability to 1.0 for the overridden
# class. Reproduced here in simplified form — the original also restricted
# this to specific follow-up-clinic note types and required the note to fall
# within 30 days of a structured follow-up date, neither of which is
# available with prophet's minimal note schema (id/note/date only). This
# version scans whatever note text is provided for a given patient-day
# instead of gating by note type.

_DEATH_PHRASES = [
    'notice of death', 'death note', 'patient deceased', 'patient is deceased',
    'patient is dead', 'deceased patient', 'physician deceased', 'report of death',
    'patient passed away', 'patient died', 'patient has died', 'patient expired',
    'patient is expired', 'patient as deceased', 'time of death', 'death was pronounced',
]

# The structured "Comprehensive Stroke Center" note template spells out the
# full mRS rubric text next to the selected grade; only checked when that
# template marker is present, to avoid false positives on ordinary narrative
# text that happens to contain similar phrasing.
_RUBRIC_PATTERNS = [
    (re.compile(r'x?\]?\s?0\s?-\s?the patient has no residual symptoms'), 0),
    (re.compile(r'x?\]?\s?1\s?-\s?the patient has no significant disability'), 1),
    (re.compile(r'x?\]?\s?2\s?-\s?the patient has slight disability'), 2),
    (re.compile(r'x?\]?\s?3\s?-\s?the patient has moderate disability'), 3),
    (re.compile(r'x?\]?\s?4\s?-\s?the patient has moderately severe disability'), 4),
    (re.compile(r'x?\]?\s?5\s?-\s?the patient has severe disability'), 5),
    (re.compile(r'x?\]?\s?6\s?-\s?the patient has expired'), 6),
]

# Narrative "Rankin is N" / "Rankin N" mentions ('mrs' -> 'rankin', matching
# the original's substitution so both wordings are caught). Only the last
# mention in the note is used, mirroring the original's assumption that later
# text reflects the final assessment. Baseline/pre-stroke mentions ("premorbid
# rankin") are stripped first so they aren't mistaken for the outcome.
_RANKIN_MENTION = re.compile(r'rankin(?:\s+scale)?(?:\s+is)?(?:\s+of)?\s*[:\-]?\s*([0-5])\b')


def _extract_note_override(raw_text: str):
    """Return an explicit mRS grade (0-6) found in the note text, or None if
    the model should fall back to its own prediction."""
    text = (raw_text or '').lower()

    for phrase in _DEATH_PHRASES:
        if phrase in text:
            return 6

    if 'comprehensive stroke center' in text:
        for pattern, grade in _RUBRIC_PATTERNS:
            if pattern.search(text):
                return grade

    narrative = text.replace('mrs', 'rankin').replace('premorbid rankin', '')
    matches = list(_RANKIN_MENTION.finditer(narrative))
    if matches:
        return int(matches[-1].group(1))

    return None


class MRSModel(_BaseModel):
    path = importlib.resources.files("prophet.models.mrs").joinpath("config.yaml")
    if path.exists():
        DEFAULT_CONFIG_PATH = str(path)
    else:
        raise FileNotFoundError(f"Configuration file not found: {path}")

    def __init__(self, config_path=None, external_ray=False):
        if config_path is None:
            config_path = self.DEFAULT_CONFIG_PATH
        super().__init__(config_path)
        self.external_ray = external_ray
        self.threshold = self.config.get('parameters', {}).get('threshold', 0.452)

        # Paths stored for lazy loading — artifacts not loaded until first prediction
        self._vectorizer = None
        self._variance_filter = None

        for resource in ('tokenizers/punkt', 'tokenizers/punkt_tab'):
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.download('punkt_tab')
                break

    @property
    def vectorizer(self):
        if self._vectorizer is None:
            pkg = importlib.resources.files("prophet.models.mrs")
            self._vectorizer = joblib.load(str(pkg.joinpath(self.config['vectorizer_path'])))
        return self._vectorizer

    @vectorizer.setter
    def vectorizer(self, value):
        self._vectorizer = value

    @property
    def variance_filter(self):
        if self._variance_filter is None:
            pkg = importlib.resources.files("prophet.models.mrs")
            self._variance_filter = joblib.load(str(pkg.joinpath(self.config['variance_filter_path'])))
        return self._variance_filter

    @variance_filter.setter
    def variance_filter(self, value):
        self._variance_filter = value

    def load_model(self, model_path: str):
        logger.info(f"Loading model from {model_path}")
        path = importlib.resources.files("prophet.models.mrs").joinpath(model_path)
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

        # Concatenate notes from the same patient on the same day
        note_df = data['note']
        note_df = (
            note_df.group_by(['id', 'date'])
            .agg(pl.col('note').str.concat(' ').alias('note'))
            .sort(['id', 'date'])
        )
        note_df = note_df.with_row_index()

        feat = note_df.select(['index', 'id', 'date'])
        if len(feat) == 0:
            raise ValueError("No notes found in the provided data.")
        logger.info(f"Generating features for n={len(feat)} patient-days (after same-day concatenation)")

        note_feat = self._preprocess_notes(note_df, feat, show_progress)
        # Carry the raw (concatenated, unstemmed) note text through so predict()
        # can run the rule-based extraction override on it.
        note_feat = note_feat.join(note_df.select(['index', 'note']), on='index', how='left')
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
            temp_dir = Path(tempfile.mkdtemp(prefix="mrs_processing_"))

            if show_progress:
                from tqdm import tqdm
                progress_bar = tqdm(total=total_notes, desc='Processing mRS notes')
            else:
                progress_bar = None

            @ray.remote(num_cpus=1)
            class MRSProcessor:
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
                            batch_results.append({
                                'index': item['index'],
                                'processed': processed,
                            })

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
            processors = [MRSProcessor.remote() for _ in range(max_concurrent_actors)]
            notes_dicts = note_df.select(['index', 'note']).to_dicts()
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

            processed_df = pl.read_parquet(parquet_files)
            # Join back with feat to get id/date, keep ordering
            processed_df = feat.join(processed_df, on='index', how='left')
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return processed_df

    def predict(self, feat: pl.DataFrame) -> pl.DataFrame:
        _start = time.time()
        logger.info(f"Prediction started at {datetime.now().strftime('%H:%M:%S')}")

        # Get processed text, apply vectorizer + variance filter
        texts = feat['processed'].fill_null('').to_list()
        X = self.vectorizer.transform(texts)
        X = self.variance_filter.transform(X)

        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        # Rule-based override: use an explicitly documented mRS grade instead
        # of the model's prediction when the note states one, forcing 100%
        # confidence on the overridden class (see _extract_note_override).
        raw_notes = feat['note'].fill_null('').to_list()
        n_overridden = 0
        for i, note_text in enumerate(raw_notes):
            override = _extract_note_override(note_text)
            if override is not None:
                preds[i] = 1 if override >= 3 else 0
                probs[i] = 1.0 if preds[i] == 1 else 0.0
                n_overridden += 1
        if n_overridden:
            logger.info(f"Rule-based extraction overrode {n_overridden}/{len(raw_notes)} predictions")

        pred = feat.select(['index', 'id', 'date']).with_columns([
            pl.Series('prob_poor', probs),
            pl.Series('prediction', preds),
        ])

        logger.info(f"Prediction finished in {time.time() - _start:.2f}s")
        return pred
