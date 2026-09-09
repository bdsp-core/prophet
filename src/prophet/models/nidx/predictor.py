from ...base._basemodel import _BaseModel
import logging
from typing import Dict, Union, Tuple
import polars as pl
import re
import string
import ray
import xgboost as xgb
import joblib
import time
import gc
import os
import shutil
import tempfile
import psutil
from pathlib import Path
from datetime import datetime
import importlib.resources
import nltk
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# The 342 n-gram features used by the trained model, in the exact order used during training.
NIDX_FEATURES = [
    'abscess', 'acyclovir', 'acyclovir zovirax', 'alert', 'alert oriented',
    'altered mental', 'amoxicillin', 'amphotericin', 'ampicillin', 'antibiotic',
    'antiviral', 'aphasia', 'aseptic', 'aseptic meningitis', 'aspirin',
    'asymmetry', 'ativan', 'autoimmune', 'bacteremia', 'bacterial',
    'bacterial meningitis', 'band', 'bcx', 'biktarvy', 'biopsy', 'bland',
    'blast', 'blot', 'blurry', 'brain', 'brain biopsy', 'brain mri',
    'ceftriaxone', 'cell count', 'cell count protein', 'cell count tube',
    'cell lymphocyte', 'cerebral', 'cerebritis', 'cervical', 'cmv', 'cns',
    'cns infection', 'cns lyme', 'collection specimen', 'colorless colorless csf',
    'communicating hydrocephalus', 'confusion', 'contrast', 'count protein',
    'count tube', 'cranial', 'critical', 'crypto', 'cryptococcal',
    'cryptococcal meningitis', 'cryptococcus', 'csf', 'csf abnormal', 'csf csf',
    'csf culture', 'csf protein', 'csf wbc', 'csfglu', 'cta', 'ctx', 'culture',
    'culture csf', 'culture negative', 'culturesmear', 'cytology', 'cytometry',
    'decadron', 'demyelinating', 'depakote', 'dexamethasone', 'diffuse',
    'diplopia', 'dizziness', 'dizziness headache', 'dna', 'doxycycline',
    'drainage', 'dysfunction', 'ebv', 'edema', 'eeg', 'elevated protein',
    'empiric', 'empirically', 'encephalitis', 'encephalitis human',
    'encephalitis human herpes', 'encephalopathy', 'enhancement', 'enterovirus',
    'epilepsy', 'evd', 'febrile', 'fever', 'fever altered', 'fever chill',
    'fever headache', 'flagyl', 'flair', 'fluconazole', 'fungal', 'gait',
    'gentamicin', 'gram stain', 'growth', 'hallucination', 'hcv', 'headache',
    'headache dizziness', 'headache fever', 'headache fever chill', 'headache lp',
    'headache negative dizziness', 'headache pain', 'headache worse', 'healthy',
    'hearing', 'hemorrhage', 'herpes', 'herpes simplex', 'hiv', 'hive',
    'hospitalization', 'hsv', 'hsv encephalitis', 'hsv pcr', 'hsv pcr csf',
    'human herpes', 'hydrocephalus', 'ibuprofen', 'icp', 'igg positive',
    'imaging lumbar', 'imaging mri', 'immediately', 'immunocompromised',
    'immunodeficiency virus hiv', 'impaired', 'impairment', 'infarct',
    'infarction', 'infection', 'infectious', 'inflammatory', 'influenza',
    'intracranial', 'ischemic', 'jcv', 'keppra', 'keppra seizure', 'laterality',
    'leptomeningeal', 'leptomeningeal enhancement', 'leptomeningeal enhancement mri',
    'leukocytosis', 'levetiracetam', 'levofloxacin', 'levothyroxine', 'lidocaine',
    'lightheadedness', 'limbic', 'listeria', 'lobe', 'localizing', 'lorazepam',
    'lowgrade', 'lp', 'lp csf', 'lp positive', 'lp wbc', 'lumbar', 'lyme',
    'lyme meningitis', 'lymphocyte', 'lymphocytic', 'lymphocytic pleocytosis',
    'malignancy', 'mass', 'mass effect', 'meningitis', 'meningitis hsv',
    'meningococcal', 'meningoencephalitis', 'meropenem', 'metastatic',
    'microbiology', 'migraine', 'migraine headache', 'mollarets', 'morphine',
    'mri', 'mri brain', 'myalgia', 'nafcillin', 'nausea', 'nausea vomiting',
    'nausea vomiting pain', 'nauseavomiting', 'neck', 'neck pain', 'neck stiffness',
    'neck stiffness pain', 'neck supple', 'negative csf', 'negative dizziness',
    'negative ebv', 'negative hsv', 'negative lyme', 'negative pain',
    'negative rash', 'neuro', 'neuroborreliosis', 'neurologic', 'neurological',
    'neurologist', 'neurology', 'neurosurgery', 'neurosyphilis', 'neuts',
    'neuts mono eos', 'nile', 'nmda', 'normal csf', 'normal limit',
    'normal protein', 'nortriptyline', 'nuchal', 'nucleated cell', 'numbness',
    'occipital', 'oligoclonal', 'oligoclonal band', 'oncology', 'opening',
    'opening csf', 'optic', 'oxycodone', 'pain headache', 'pain nausea vomiting',
    'pain negative', 'pain positive', 'paraneoplastic', 'paresthesia', 'pcr',
    'pcr csf', 'pcr negative', 'penicillin', 'pet', 'phenobarbital',
    'photophobia', 'photophobia pain', 'pleocytosis', 'pml', 'polymicrobial',
    'positional', 'positive headache', 'positive lyme', 'positive nausea',
    'positive rpr', 'postoperative', 'powassan', 'prednisone',
    'progressive multifocal', 'radiculopathy', 'rash', 'read', 'reflex',
    'rifampin', 'rigidity', 'rituximab', 'rocephin', 'rpr', 'rpr titer',
    'seizure', 'sepsis', 'serology negative', 'severity', 'shingle', 'shock',
    'smear', 'solumedrol', 'specimen cerebrospinal', 'spinal', 'spine', 'steroid',
    'stiff', 'stiffness', 'stroke', 'subarachnoid hemorrhage', 'swallowing',
    'syphilis', 'syphilis positive', 'tap', 'taper', 'tia', 'tick', 'tingling',
    'tissue', 'titer', 'toxoplasmosis', 'transplant', 'tremor', 'treponemal',
    'trimethoprim', 'tspot', 'valacyclovir', 'valacyclovir valtrex', 'vanc',
    'vancocin', 'vancomycin', 'varicella', 'vdrl', 'ventriculitis', 'vertigo',
    'vimpat', 'viral encephalitis', 'viral load', 'viral meningitis', 'vomiting',
    'vomiting pain', 'vzv', 'vzv encephalitis', 'wbc', 'wbc absolute',
    'wbc csfglu', 'wbc wbc', 'weakness', 'west', 'west nile', 'wnl',
    'wordfinding', 'xanax', 'xanthochromia', 'zofran', 'zoster', 'zosyn',
]

# Custom stopwords added on top of NLTK English stopwords during training.
# "no" and "without" are explicitly kept (removed from stopwords) to preserve negation.
_CUSTOM_STOPWORDS = set("""
solution suggest initiation nih aki stop peg ivf ivbmbp nrbc came young might swab early added
syndrome closure rather previously troponin sample initial location tmax syringe covid gait
especially blast esr pcp event life later size error unfortunately essential eval spot angio
clinically slight palpation provider visit impression slightly unable related drug dyspnea sinus
three similar first already pertinent dysuria ent come calcium thank sent one got abdominal
reaction ace cardiovascular cough left clinical hypertension diarrhea loss intravenous mlhr
physical meei picc june diff returned resolved temp factor creatinine includes rue mgkg intake
partner crbc hyperlipidemia subsequent making ivpb lue study disposition cva tid soon favor hent
muscle pursue social motoprolol unit voice inject facilityadministered called sweat admit kidney
intermittent sob chief fine pregnancy method nerve albumin alt appetite bleeding bowel case
chronic close cold complete considered constipation diagnosed differential difficulty disorder done
drop electrolyte emergency episode etiology fatigue final fully gdl improved injection inpatient
insomnia intact line management mul night nightly noticed nurse old outpatient palpitation
potassium presenting procedure rare screen speech spoke surgical suspected twelve uri vision
vitals weight workup yesterday arrival atrial transitioned routine tenderness recommendation
associated term earlier incontinence crp away spent room joint placed except multiple small
psychiatricbehavioral motion took onset rdw issue standing asthma condition andor red allergen
renal reflux twice setting woman urinary continuous think kul please egfr sore florence
metoprolol comprehensive end severe make mpv head michael dr illness secondary describes status
visual disturbance unspecified ear much flp include remains strength memory rate resp site stable
instructed scheduled lower free minute needed feasible fluoride feat side recommended vein liver
mrn benign going gastrointestinal ion given cal axone secondary consistent date abnormality osa
serum concern also and as cal change consistent cus cycles d1 daily dat dated day days december
deer definitive definitively delayed delays demonstrat due for have his in likely no not of on or
puncture right such to was with would wout assessment disease complaint anxiety persistent seen
consider month past possible history since around leg following likely evaluation evaluated mgdl
reviewed symptom sleep pending evidence index exam lymph diagnosis musculoskeletal given last
admitted concerning present new otherwise result interval dose taking without testing lab
treatment medical mild system report therapy note time gastroesophageal chest prior transferred
treated course show wife continue though male blood continues received finding consult primary
back fluid followup started intensity recurrent unclear lung return able take help found ankle
presented recovery htn uti presentation primarily reason admission intervention bid consulted
flow shortness manual hepatitis slp ache frequency duration mgh discharged md started possible
iv fu last plt panel overall followup reason taking course medication cb cf continue index plan
datetime dr hpi known detected antigen allergy mgdl pt well hct yo female history am pm able
active activity alb alcohol alkp assessment assessmentplan bun ca car cbc cl co coags component
cre data dbili degenerative diabetes disc discussed disease drive estimated eye follow gfr glu
hgb hypercholesterolemia input inr iso lab list mellitus mg ocular patient pdmp problem ptt
recent refill result reviewed sonopalpation sp tablet tbili thickening thoracic tp tsh type ucre
value vitamin walk walker within respiratory amlodipine speak lfts morning bundle compliant foot
acceptable nasal heart rehab increase regarding stopped underwent surgery march however yet gtt
bolus felt home patient tylenol oral still chloride glycol arthroscopy still see due additional
seen notable appointment use including order like need person would none otherwise matter agree
tried also could today current new past may take home appears ago given better good change return
detail question care starting family since followed previous back presentation general primarily
based ref saw thought raising performed without daily month week still see due additional seen
notable appointment use including order like need person would none otherwise matter agree tried
obtain date high feeling day per denies note symptom using currently name lab prn continues sign
however state went patient basic am finding son code also could today current new past may take
home appears ago given better good change return detail question care starting family since
followed previous back presentation general primarily based ref saw thought raising htn get right
low file work clear without note initially range male year several full feel check control
department completed recommend ongoing obtained level though service state regarding about obtain
date high feeling day per denies symptom using currently name lab prn continues sign however
state went patient basic am finding son code also could today current new past may take home
appears ago given better good change return detail question care starting family since followed
previous back presentation general primarily based ref saw thought raising total skin showed
placement unknown every mood treatment month system discus baseline start record medication day
repeat sig developed blood mouth med le chloride musculoskeletal transfer murmur presented urine
noted source cause two medical pressure respiratory fall hospital question comment center
pneumonia sodium rbc reported encounter note allergy outside risk glucose cancer review infusion
pulse discharge affected dose limited lisinopril improving anxiety clinic consulted regimen
consult april oral extremity transferred involvement test continued senna service state
subjective hour bwh rehab pending mdm found capsule osh felt patient weekly attending october
bruising dispense finding primary also bwh continues current future given home htn may note
ordered past surgical post rehab take time today transferred
""".split())

# Keep negation words that were explicitly removed from stopwords during training
_CUSTOM_STOPWORDS.discard('no')
_CUSTOM_STOPWORDS.discard('without')


def _preprocess_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """Replicates the preprocess_text() function used during NIDX model training."""
    text = text.lower()
    text = ''.join(char for char in text if ord(char) < 128)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = ''.join([char for char in text if char not in string.punctuation])
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    lemmatized = [
        lemmatizer.lemmatize(w) for w in words
        if w not in stop_words and (len(w) > 2 or w == 'lp')
    ]
    lemmatized = [w for w in lemmatized if w not in stop_words and (len(w) > 2 or w == 'lp')]
    result = ' '.join(lemmatized).strip()
    return result if result else ' '


def _extract_features(processed_text: str) -> dict:
    """Count occurrences of each NIDX feature n-gram in preprocessed text."""
    result = {}
    for feat in NIDX_FEATURES:
        # Use word-boundary-aware matching: feature must be surrounded by spaces or text boundaries
        count = len(re.findall(r'(?<![a-z])' + re.escape(feat) + r'(?![a-z])', processed_text))
        result[feat] = count
    return result


class NIDXModel(_BaseModel):
    path = importlib.resources.files("prophet.models.nidx").joinpath("config.yaml")
    if path.exists():
        DEFAULT_CONFIG_PATH = str(path)
    else:
        raise FileNotFoundError(f"Configuration file not found: {path}")

    def __init__(self, config_path=None, external_ray=False):
        if config_path is None:
            config_path = self.DEFAULT_CONFIG_PATH
        super().__init__(config_path)
        self.external_ray = external_ray

        for resource in ('corpora/wordnet', 'corpora/stopwords'):
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.download(resource.split('/')[-1])

    def load_model(self, model_path: str):
        logger.info(f"Loading model from {model_path}")
        path = importlib.resources.files("prophet.models.nidx").joinpath(model_path)
        booster = xgb.Booster()
        booster.load_model(str(path))
        return booster

    def run(self, data: Dict[str, pl.DataFrame], show_progress=False, return_features=False, force_casting=False) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
        feat = self.preprocess(data, show_progress, force_casting)
        pred = self.predict(feat)
        if return_features:
            return feat, pred
        return pred

    def preprocess(self, data: Dict[str, pl.DataFrame], show_progress=False, force_casting=False) -> pl.DataFrame:
        logger.info(f"Preprocessing started at {datetime.now()}")
        data = super().preprocess(data, show_progress, force_casting)
        data['note'] = data['note'].with_row_index()

        feat = data['note'].select(['index', 'id'])
        if len(feat) == 0:
            raise ValueError("No notes found in the provided data.")
        logger.info(f"Generating features for n={len(feat)} notes")

        note_feat = self._preprocess_notes(data['note'], feat, show_progress)
        logger.info(f"Preprocessing finished at {datetime.now()}")
        return note_feat

    def _preprocess_notes(self, note_df: pl.DataFrame, feat: pl.DataFrame, show_progress=False) -> pl.DataFrame:
        from nltk.corpus import stopwords as nltk_stopwords

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
            temp_dir = Path(tempfile.mkdtemp(prefix="nidx_processing_"))

            if show_progress:
                from tqdm import tqdm
                progress_bar = tqdm(total=total_notes, desc='Processing NIDX notes')
            else:
                progress_bar = None

            custom_stopwords = _CUSTOM_STOPWORDS

            @ray.remote(num_cpus=1)
            class NIDXProcessor:
                def __init__(self):
                    import nltk
                    from nltk.stem import WordNetLemmatizer
                    from nltk.corpus import stopwords as nltk_sw
                    self.lemmatizer = WordNetLemmatizer()
                    sw = set(nltk_sw.words('english'))
                    sw.update(custom_stopwords)
                    sw.discard('no')
                    sw.discard('without')
                    self.stop_words = sw

                def process_batch(self, batch_data, batch_id, temp_dir_str):
                    batch_results = []
                    try:
                        for item in batch_data:
                            text = item['note'] or ''
                            processed = _preprocess_text(text, self.lemmatizer, self.stop_words)
                            row = {'index': item['index']}
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
            processors = [NIDXProcessor.remote() for _ in range(max_concurrent_actors)]
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

            note_feat = pl.read_parquet(parquet_files)
            note_feat = note_feat.join(note_df.select(['index', 'id']), on='index', how='left')
            note_feat = feat.join(note_feat, on=['index', 'id'], how='left', validate='1:1').fill_null(0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return note_feat

    def predict(self, feat: pl.DataFrame) -> pl.DataFrame:
        _start = time.time()
        logger.info(f"Prediction started at {datetime.now().strftime('%H:%M:%S')}")

        X = feat.select(NIDX_FEATURES).to_pandas()
        dmatrix = xgb.DMatrix(X, feature_names=NIDX_FEATURES)
        prob_yes = self.model.predict(dmatrix)

        threshold = self.config['parameters']['threshold']
        logger.info(f'Using threshold of {threshold}')

        pred = feat.select(['index', 'id']).with_columns([
            pl.Series('prob_NO', 1 - prob_yes),
            pl.Series('prob_YES', prob_yes),
            pl.Series('prediction', (prob_yes > threshold).astype(int)),
        ])

        logger.info(f"Prediction finished in {time.time() - _start:.2f}s")
        return pred
