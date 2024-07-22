from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import ray

def build_feat_mat(data, vocab):
    icds = data['icd_codes']
    meds = data['medications']
    notes = data['notes']

    icd_vocab = vocab['icd_codes']
    med_vocab = vocab['medications']
    notes_vocab = vocab['notes']

    # set time range
    time_range = pd.DateOffset(months=18)

    # function for BoW extraction from notes
    def get_BoW_features_from_notes(notes, vocabulary):
        bow_vectorizer = CountVectorizer(vocabulary= vocabulary)  # create a BoW feature extractor
        bow_vectorizer.fit(notes)
        X = bow_vectorizer.transform(notes)   # get features matrix X
        X = X.toarray()    # initially X is a sparse matrix, you can convert it to a non-sparse numpy array
    
        # The resulting X is
        # array([[1, 0],    # the first note has 1 occurrence of ‘first’, and 0 occurrence of ‘second’
        #             [0, 1],    # the second note has 0 occurrence of ‘first’, and 1 occurrence of ‘second’
        #             [0, 0]],    # the third note has 0 occurrence of ‘first’, and 0 occurrence of ‘second’
        # dtype=int64)
    
        return X

    # stemmer initialization
    stemmer = PorterStemmer()
    
    # ray initialization
    ray.init()
    icd_ray = ray.put(icds)
    med_ray = ray.put(meds)
    notes_ray = ray.put(notes)
    icd_vocab_ray = ray.put(icd_vocab)
    med_vocab_ray = ray.put(med_vocab)
    notes_vocab_ray = ray.put(notes_vocab)

    @ray.remote
    def loop(i, icd_ray, med_ray, notes_ray, icd_vocab_ray, med_vocab_ray, notes_vocab_ray):
        pid = notes_ray.bdsp_patient_id.iloc[i]
        note = notes_ray.note.iloc[i]
        date_note = notes_ray.date_note.iloc[i]
        date_note_min = date_note - time_range
        date_note_max = date_note + time_range

        # get icds in window, create feature matrix
        icds = icd_ray[(icd_ray.bdsp_patient_id == pid) & (icd_ray.date_icd >= date_note_min) & (icd_ray.date_icd <= date_note_max)].values
        icds = [str(x) for x in icds.tolist()]
        icds = ' '.join(icds)
        feat_icds = np.array([int(re.search(r'(?:{})'.format(re.escape(x)), icds, re.IGNORECASE) is not None) for x in icd_vocab_ray]).reshape(1,len(icd_vocab_ray))
        has_icd = feat_icds.max()

        # get meds in window, create feature matrix
        meds = med_ray.loc[
            (med_ray.bdsp_patient_id == pid) &
            (
                (med_ray.date_med_start <= date_note_max) | 
                (med_ray.date_med_end >= date_note_min)
            ),
            'med'
        ]
        meds = [str(x) for x in meds.tolist()]
        meds = ' '.join(meds)
        feat_meds = np.array([int(re.search(r'\b{}\b'.format(re.escape(x)), meds, re.IGNORECASE) is not None) for x in med_vocab_ray]).reshape(1,len(med_vocab_ray))
        has_med = feat_meds.max()

        # stem the note, create feature matrix
        X_list = np.zeros(len(notes_vocab_ray))
        sentences = note.split('.')
        for sentence in sentences:
            words = word_tokenize(sentence)
            stemmed_words = [stemmer.stem(word.lower()) for word in words]
            array = []
            for word in notes_vocab_ray:
                tokens = word_tokenize(word)
                vocab = [stemmer.stem(token.lower()) for token in tokens]
                feat_notes_iter = get_BoW_features_from_notes([" ".join(stemmed_words)], vocab)
                for i in range(len(tokens)):
                    if tokens[i] == '[' or tokens[i] == ']' or tokens[i]== '(' or tokens[i]== ')':
                        result_count = sentence.count(tokens[i])
                        if result_count > 0:
                            feat_notes_iter[0,i] = result_count
                value = np.min(feat_notes_iter)
                array.append(value.astype(int))
            X_list += array
        X_list = X_list.astype(int)
        feat_notes = np.reshape(X_list, (1, len(notes_vocab_ray)))

        # put feature matrix together and return
        result = np.concatenate([feat_icds, feat_meds, feat_notes], axis = 1)
        return (pid, date_note, note, has_icd, has_med, result)

    # run ray loop
    length = len(notes)
    feats = ray.get([loop.remote(i, icd_ray, med_ray, notes_ray, icd_vocab_ray, med_vocab_ray, notes_vocab_ray) for i in range(length)])

    # ray shutdown
    ray.shutdown()

    # Unpack the results
    pids, dates, notes, has_icds, has_meds, feat_vectors = zip(*feats)

    # build feature matrix dataframe
    feat_mat = np.array(feat_vectors)
    feat_mat_2 = np.reshape(feat_mat, (length, (len(icd_vocab) + len(med_vocab) + len(notes_vocab))))
    col_names = icd_vocab + med_vocab + notes_vocab
    df_feat = pd.DataFrame(feat_mat_2, columns=col_names)

    df_iden = pd.DataFrame({
        'bdsp_patient_id', pids,
        'date_note', dates,
        'note', notes,
        'icd+', has_icds,
        'med+', has_meds
    })

    return df_iden, df_feat