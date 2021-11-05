import pandas as pd
import numpy as np
from numpy.linalg import norm
import argparse
import os
from glob import glob

from scipy import stats
from scipy import sparse
from scipy.sparse import linalg
from scipy.spatial import procrustes

from preprocess import preprocess
from gensim.models import Word2Vec

from collections import Counter

import pickle

def aligne_mtx(a,b):
    mtx1, mtx2, disparity = procrustes(a, b)
    return mtx1, mtx2


def get_data(name):
    """
    Loading data into pandas DataFrame
    :param name: name of .csv to fetch
    :return: pandas.DataFrame
    """
    data = pd.read_csv(name,sep="\t")
    print("Length of DataFrame: %d"%len(data))
    return data


def initialize_model(df, dir_name, arr_dir_name, model_type,is_init=True):
    """
    Save a "init.model" corresponding to the first embedding space
    :param df: bunch of documents in DataFrame
           dir_name: name of directory to save models
           arr_dir_name: name of directory to save arrays
           model_type: either SVD or SGNS
    :return: .model
    :return: vocab: list of words
    :return: frequency: dict of frequency by token from preprocess
    """
    # Creating Folder for storing models
    if not os.path.isdir(dir_name):
        print("Creating Models Directory...")
        os.makedirs(dir_name)
    print("Models Saving Directory: %s"%dir_name)

    # Creating Folder for storing models
    if not os.path.isdir(arr_dir_name):
        print("Creating Array Directory...")
        os.makedirs(arr_dir_name)
    print("Array Saving Directory: %s"%arr_dir_name)
    # Initialize model
    model_list, frequency, vocab = modeling(df,model_type, is_init)
    model_name = dir_name + "/init_"+ model_type
    if model_type == "SGNS":
        model = model_list
        model.save(model_name + ".model")
    elif model_type == "SVD":
        model = model_list

        np.save(model_name + ".npy", model_list[0])

        f = open(model_name + "_indx2tok.pkl", "wb")
        t = open(model_name + "_vocab.pkl", "wb")
        pickle.dump(model_list[1], f)
        pickle.dump(model_list[2], t)
        f.close()
        t.close()

        out_wcounts = open(model_name + "_wcounts.pkl", "wb")
        out_ucounts = open(model_name + "_ucounts.pkl", "wb")
        out_sgcounts = open(model_name + "_sgcounts.pkl", "wb")

        pickle.dump(model_list[5], out_wcounts)
        pickle.dump(model_list[3], out_ucounts)
        pickle.dump(model_list[4], out_sgcounts)

        out_wcounts.close()
        out_ucounts.close()
        out_sgcounts.close()
    return model, vocab, frequency


def svd_init(headlines):
    """
    Initialize first word count matrix
    :param headlines: list of docs = sentences
    :return:
    """
    unigram_counts = Counter()
    for ii, headline in enumerate(headlines):
        for token in headline:
            unigram_counts[token] += 1

    tok2indx = {tok: indx for indx, tok in enumerate(unigram_counts.keys())}
    indx2tok = {indx: tok for tok, indx in tok2indx.items()}
    print('vocabulary size: {}'.format(len(unigram_counts)))

    back_window = 5
    front_window = 5
    skipgram_counts = Counter()
    for iheadline, headline in enumerate(headlines):
        tokens = [tok2indx[tok] for tok in headline]
        for ii_word, word in enumerate(tokens):
            ii_context_min = max(0, ii_word - back_window)
            ii_context_max = min(len(headline) - 1, ii_word + front_window)
            ii_contexts = [
                ii for ii in range(ii_context_min, ii_context_max + 1)
                if ii != ii_word]
            for ii_context in ii_contexts:
                skipgram = (tokens[ii_word], tokens[ii_context])
                skipgram_counts[skipgram] += 1

    row_indxs = []
    col_indxs = []
    dat_values = []
    ii = 0
    for (tok1, tok2), sg_count in skipgram_counts.items():
        ii += 1
        row_indxs.append(tok1)
        col_indxs.append(tok2)
        dat_values.append(sg_count)
    wwcnt_mat = sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))

    return unigram_counts, tok2indx, indx2tok, skipgram_counts, wwcnt_mat


def update_wwcnt_matrix(unigram_counts, skipgram_counts, toks, headlines):
    """
    :param unigram_counts: old unigram counts to update
    :param skipgram_counts: old skipram_counts to update
    :param toks: list of indx2tok and tok2indx
    :param headlines: new documents
    :return: updated ucounts, updated sgcounts, updated wwcnt matrix
    """
    indx2tok = toks[0]
    tok2indx = toks[1]

    for ii, headline in enumerate(headlines):
        for token in headline:
            if token in unigram_counts:
                unigram_counts[token] += 1

    back_window = 5
    front_window = 5
    for iheadline, headline in enumerate(headlines):
        tokens = [tok2indx[tok] for tok in headline if tok in unigram_counts]
        for ii_word, word in enumerate(tokens):
            ii_context_min = max(0, ii_word - back_window)
            ii_context_max = min(len(tokens) - 1, ii_word + front_window)
            ii_contexts = [
                ii for ii in range(ii_context_min, ii_context_max + 1)
                if ii != ii_word]
            for ii_context in ii_contexts:
                skipgram = (tokens[ii_word], tokens[ii_context])
                skipgram_counts[skipgram] += 1

    row_indxs = []
    col_indxs = []
    dat_values = []
    ii = 0
    for (tok1, tok2), sg_count in skipgram_counts.items():
        ii += 1
        row_indxs.append(tok1)
        col_indxs.append(tok2)
        dat_values.append(sg_count)
    wwcnt_mat = sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))

    return unigram_counts, skipgram_counts, wwcnt_mat


def update_modeling(df, dir_name, date_name, prev_model, model_type, prev_ucounts = None, prev_sgcounts = None, toks = None, is_init = False):
    """
    Save a "date.model" corresponding to the updating of previous model
    :param df: bunch of document to use for updating
    :param dir_name: directory to save new model
    :param date_name: name for new model
    :param prev_model: previous embedding model
    :param prev_ucounts: previous unigram counts, defined only if SVD
    :param prev_sgcounts: previous skip gram counts, defined only if SVD
    :param toks: list of indx2tok, tok2indx
    :return: new_model
    :return: frequency: dict of frequency by token from
    :return: vocab: list of words
    """
    if model_type == "SGNS":
        name = dir_name + "/" + str(date_name) + ".model"
    elif model_type=="SVD":
        name = dir_name + "/" + str(date_name) + ".npy"
    print("Cleaning data at: %s..."%date_name)
    sentences, frequency = preprocess(df,is_init= is_init, lang=LANGUAGE)
    print("Updating model with %d documents"%len(sentences))
    if model_type == "SGNS":
        prev_model.train(sentences,epochs=prev_model.iter, total_examples=prev_model.corpus_count)
        prev_model.save(name)
        vocab = list(prev_model.wv.vocab.keys())
    elif model_type == "SVD":
        new_ucounts, new_sgcounts, new_wwcnt = update_wwcnt_matrix(prev_ucounts, prev_sgcounts, toks, sentences)
        vocab = list(toks[1].keys())
        pmi_mat, ppmi_mat, spmi_mat, sppmi_mat = ppmi_matrices(new_wwcnt, new_sgcounts)
        uu, ss, vv, word_vecs, word_vecs_norm = svd_modeling(ppmi_mat, embedding_size=100)
        prev_model = word_vecs_norm, toks[0], toks[1], new_ucounts, new_sgcounts, new_wwcnt

        np.save(name, prev_model[0])

        indx2tok_in = open(dir_name + "/init_SVD_indx2tok.pkl", "wb")
        vocab_in = open(dir_name + "/init_SVD_vocab.pkl", "wb")
        ucounts_in = open(dir_name + "/init_SVD_ucounts.pkl", "wb")
        sgcounts_in = open(dir_name + "/init_SVD_sgcounts.pkl", "wb")
        wcounts_in = open(dir_name + "/init_SVD_wcounts.pkl", "wb")

        pickle.dump(prev_model[1], indx2tok_in)
        pickle.dump(prev_model[2], vocab_in)
        pickle.dump(prev_model[3], ucounts_in)
        pickle.dump(prev_model[4], sgcounts_in)
        pickle.dump(prev_model[5], wcounts_in)

        indx2tok_in.close()
        vocab_in.close()
        ucounts_in.close()
        sgcounts_in.close()
        wcounts_in.close()

    return prev_model, frequency, vocab


def modeling(df, model_type, is_init=False):
    """
    Gensim Word2Vec or SVD modeling
    :param df: documents in DataFrame
    :param model_type: SGNS or SVD
    :return: .model
    :return: frequency: dict of frequency by token from preprocess
    """
    print("Cleaning Data...")
    sentences, frequency = preprocess(df, is_init,lang=LANGUAGE)
    print(len(sentences))
    print("Modeling...")
    if model_type == "SGNS":
        model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4, iter=20, sg=1)
        vocab = list(model.wv.vocab.keys())
    elif model_type == "SVD":
        unigram_counts, tok2indx, indx2tok, skipgram_counts, wwcnt_mat = svd_init(sentences)
        vocab = list(tok2indx.keys())
        pmi_mat_init, ppmi_mat_init, spmi_mat_init, sppmi_mat_init = ppmi_matrices(wwcnt_mat,skipgram_counts)
        uu, ss, vv, word_vecs, word_vecs_norm = svd_modeling(ppmi_mat_init, embedding_size=100)
        model = word_vecs_norm, indx2tok, tok2indx, unigram_counts, skipgram_counts, wwcnt_mat
    else:
        raise ValueError('Model Type Should be SGNS or SVD, you chose:' + str(model_type))
    return model, frequency, vocab


def svd_modeling(mat, embedding_size):
    """
    Compute SVD from matrix
    :param mat: PPMI or SPPMI matrix
    :param embedding_size: size of word embedding at the end
    :return: all the computed matrices
    """

    uu, ss, vv = linalg.svds(mat, embedding_size)
    word_vecs = uu + vv.T
    word_vecs_norm = word_vecs / np.sqrt(np.sum(word_vecs * word_vecs, axis=1, keepdims=True))
    return uu, ss, vv, word_vecs, word_vecs_norm


def ppmi_matrices(wwcnt_mat, skipgram_counts):
    """
    Compute PPMI or SPPMI from word count matrix
    :param wwcnt_mat: word count matric
    :param skipgram_counts:
    :return:
    """
    num_skipgrams = wwcnt_mat.sum()
    assert (sum(skipgram_counts.values()) == num_skipgrams)

    # for creating sparse matrices
    row_indxs = []
    col_indxs = []

    pmi_dat_values = []  # pointwise mutual information
    ppmi_dat_values = []  # positive pointwise mutial information
    spmi_dat_values = []  # smoothed pointwise mutual information
    sppmi_dat_values = []  # smoothed positive pointwise mutual information

    sum_over_words = np.array(wwcnt_mat.sum(axis=0)).flatten()
    sum_over_contexts = np.array(wwcnt_mat.sum(axis=1)).flatten()

    # smoothing
    alpha = 0.75
    sum_over_words_alpha = sum_over_words ** alpha
    nca_denom = np.sum(sum_over_words_alpha)

    ii = 0
    for (tok_word, tok_context), sg_count in skipgram_counts.items():
        ii += 1

        nwc = sg_count
        Pwc = nwc / num_skipgrams
        nw = sum_over_contexts[tok_word]
        Pw = nw / num_skipgrams
        nc = sum_over_words[tok_context]
        Pc = nc / num_skipgrams

        nca = sum_over_words_alpha[tok_context]
        Pca = nca / nca_denom

        pmi = np.log2(Pwc / (Pw * Pc))
        ppmi = max(pmi, 0)
        spmi = np.log2(Pwc / (Pw * Pca))
        sppmi = max(spmi, 0)

        row_indxs.append(tok_word)
        col_indxs.append(tok_context)
        pmi_dat_values.append(pmi)
        ppmi_dat_values.append(ppmi)
        spmi_dat_values.append(spmi)
        sppmi_dat_values.append(sppmi)

    pmi_mat = sparse.csr_matrix((pmi_dat_values, (row_indxs, col_indxs)))
    ppmi_mat = sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)))
    spmi_mat = sparse.csr_matrix((spmi_dat_values, (row_indxs, col_indxs)))
    sppmi_mat = sparse.csr_matrix((sppmi_dat_values, (row_indxs, col_indxs)))

    print('done')

    return pmi_mat, ppmi_mat, spmi_mat, sppmi_mat


def already_exist(dir_name, model_type):
    """
    Check if model has already been initialized
    :param dir_name: directory with models
    :param model_type: SVD or SGNS
    :return: True if it is already initialized
    """
    print(dir_name)
    if not os.path.isdir(dir_name):
        print("Creating Models Directory...")
        os.makedirs(dir_name)

    list_files = os.listdir(dir_name)
    if model_type == "SGNS":
        if "init_SGNS.model" in list_files:
            return True
        else:
            return False
    elif model_type == "SVD":
        if "init_SVD.npy" in list_files:
            return True
        else:
            return False


def get_last_model(dir_name, model_type):
    """
    If model has been initialized, retrieve last model to update
    :param dir_name: directory with models
    :return: .model last model, date: str "1995-01" or "init"
    """
    if model_type == "SGNS":
        print(dir_name)
        list_files = [m for m in sorted(glob(dir_name + '/*.model'))]
        list_files = [f.replace(dir_name + "\\","") for f in list_files]
        print(list_files)
        if len(list_files) > 1:
            list_files.remove("init_SGNS.model")
            model_name = list_files[-1]
        else:
            model_name = "init_SGNS.model"
        model = Word2Vec.load(dir_name + "/" + model_name)
        date = model_name.replace(".model", "")
        return model, date

    elif model_type == "SVD":
        list_to_remove = ["init_SVD_vocab.pkl", "init_SVD_ucounts.pkl", "init_SVD_sgcounts.pkl", "init_SVD_wcounts.pkl",
                          "init_SVD_indx2tok.pkl"]
        list_files = [m for m in sorted(glob(dir_name + '/*.npy')) if m not in list_to_remove]
        list_files = [f.replace(dir_name + "\\", "") for f in list_files]
        print(list_files)
        if len(list_files) > 1:
            list_files.remove(dir_name + "/" + "init_SVD.npy")
            model_name = list_files[-1]
        else:
            model_name = dir_name + "/" + "init_SVD.npy"

        model = np.load(model_name, allow_pickle=True)

        ucounts_in = open(dir_name + "/init_SVD_ucounts.pkl","rb")
        sgcounts_in = open(dir_name + "/init_SVD_sgcounts.pkl", "rb")
        wcounts_in = open(dir_name + "/init_SVD_wcounts.pkl", "rb")
        tok2indx_in = open(dir_name + "/init_SVD_vocab.pkl","rb")
        indx2tok_in = open(dir_name + "/init_SVD_indx2tok.pkl","rb")

        ucounts = pickle.load(ucounts_in)
        sgcounts = pickle.load(sgcounts_in)
        wcounts = pickle.load(wcounts_in)
        indx2tok = pickle.load(indx2tok_in)
        tok2indx = pickle.load(tok2indx_in)

        toks = indx2tok, tok2indx

        ucounts_in.close()
        sgcounts_in.close()
        wcounts_in.close()
        tok2indx_in.close()
        indx2tok_in.close()

        date = model_name.replace(".npy", "")
        date = date.replace(dir_name + "/","")
        print("date: %s"%date)

        return model, date, ucounts, sgcounts, wcounts, toks


def get_previous_model(dir_name, model_type):
    if model_type == "SGNS":
        list_files = [m for m in sorted(glob(dir_name + '/*.model'))]
        list_files = [f.replace(dir_name + "\\","") for f in list_files]
        if len(list_files) > 2:
            list_files.remove("init_SGNS.model")
            model_name = list_files[-2]
        else:
            model_name = "init_SGNS.model"
        model = Word2Vec.load(dir_name + "/" + model_name)
        return model

    elif model_type == "SVD":
        list_to_remove = ["init_SVD_vocab.pkl","init_SVD_ucounts.pkl","init_SVD_sgcounts.pkl","init_SVD_wcounts.pkl","init_SVD_indx2tok.pkl"]
        list_files = [m for m in sorted(glob(dir_name + '/*.npy')) if m not in list_to_remove]
        list_files = [f.replace(dir_name + "\\","") for f in list_files]
        print(list_files)
        if len(list_files) > 2:
            list_files.remove(dir_name + "/" + "init_SVD.npy")
            model_name = list_files[-2]
        else:
            model_name = dir_name + "/" + "init_SVD.npy"
        model = np.load(model_name)

        ucounts_in = open(dir_name + "/init_SVD_ucounts.pkl", "rb")
        sgcounts_in = open(dir_name + "/init_SVD_sgcounts.pkl", "rb")
        wcounts_in = open(dir_name + "/init_SVD_wcounts.pkl",'rb')
        tok2indx_in = open(dir_name + "/init_SVD_vocab.pkl", "rb")
        indx2tok_in = open(dir_name + "/init_SVD_indx2tok.pkl", "rb")


        ucounts = pickle.load(ucounts_in)
        sgcounts = pickle.load(sgcounts_in)
        wcounts = pickle.load(wcounts_in)
        indx2tok = pickle.load(indx2tok_in)
        tok2indx = pickle.load(tok2indx_in)

        m = model, indx2tok, tok2indx, ucounts, sgcounts, wcounts
        return m


def get_end_date(df):
    """
    Get last timestamp of our data: where to stop
    :param df: entire dataset
    :return: date : "2004-12"
    """
    list_date = sorted(list(df.mois.unique()))
    return list_date[-1]


def fetch_new_data(df, last_date, size_init):
    """
    Getting data corresponding to the new month to modelize
    :param df: entire dataset or bunch of documents arriving in streaming
    :param last_date: last modeling date
    :param size_init: number of date to initialize first model
    :return: new DataFrame,new_date
    """
    list_date = sorted(list(df.mois.unique()))
    print(last_date)
    if last_date == "init_SGNS" or last_date == "init_SVD":
        new_date = list_date[size_init]
    else:
        index_last_date = list_date.index(last_date)
        new_date = list_date[index_last_date + 1]
    print("Fetching new data at: %s"%new_date)
    new_df = df[df.mois == new_date]
    return new_df, str(new_date)


def initialize_arrays(vocab, frequency, arr_dir_name):
    """
    Initialize first columns of euclidean and frequency array and it in directory
    :param vocab: list of words
    :param frequency: DefaultDict containing frequency of each word
    :param arr_dir_name: Name of directory for storing arrays
    :return: arrays with len(col) = len(vocab)
    """
    list_init = list()
    list_freq_init = list()
    list_corr_init = list()

    for i in range(len(vocab)):
        token = vocab[i]
        list_init.append([0])
        list_corr_init.append([0])
        list_freq_init.append([frequency[token]])

    euclidean_array = np.array(list_init)
    freq_array = np.array(list_freq_init)
    corr_array = np.array(list_corr_init)

    np.save(arr_dir_name + "/euclidean_SGNS.npy", euclidean_array)
    np.save(arr_dir_name + "/euclidean_SVD.npy", euclidean_array)
    np.save(arr_dir_name + "/freq.npy", freq_array)
    np.save(arr_dir_name + "/correlation_SGNS.npy", corr_array)
    np.save(arr_dir_name + "/correlation_SVD.npy", corr_array)


def update_movement(arr_dir_name, frequency, vocab, prev_model, new_model, model_type):
    """
    Update movement and frequency arrays
    :param arr_dir_name: name of directory with stored array
    :param frequency: DefaultDict of frequency for each token
    :param vocab: list of words
    :param prev_model: previous model
    :param model: new model (.model if SGNS, list if SVD [word_vecs_norm, indx2tok, tok2indx, unigram_counts, skipgram_counts, wwcnt_mat]
    :return: new_euclidean, new_freq
    """
    prev_freq = np.load(arr_dir_name + "/freq.npy")
    prev_euclidean = np.load(arr_dir_name + "/euclidean_" + model_type + ".npy")

    print("Euclidean Array Shape: %s"%str(prev_euclidean.shape))
    print("Frequency Array Shape: %s" %str(prev_freq.shape))

    new_freq_list = list()
    new_euclidean_list = list()

    if model_type == "SVD":
        # Alignment step
        prev_mtx = prev_model[0]
        new_mtx = new_model[0]
        aligned_prev_model, aligned_new_model = aligne_mtx(prev_mtx, new_mtx)
        prev_model = aligned_prev_model
        new_model = aligned_new_model
        for token in vocab:
            new_freq_list.append((frequency[token]))
        new_freq = np.hstack((prev_freq, np.atleast_2d(np.array(new_freq_list)).T))

        vector_euclidean = np.zeros(len(vocab))
        for i in range(len(vocab)):
            a = aligned_prev_model[i]
            b = aligned_new_model[i]
            vector_euclidean[i] = norm(a - b)  # Euclidean Distance
        new_euclidean = np.hstack((prev_euclidean, np.atleast_2d(np.array(vector_euclidean)).T))


    elif model_type == "SGNS":
        for token in vocab:
            new_freq_list.append((frequency[token]))
            euclidean_distance = get_euclidean(prev_model, new_model, token)
            new_euclidean_list.append(euclidean_distance)
        new_freq = np.hstack((prev_freq, np.atleast_2d(np.array(new_freq_list)).T))
        new_euclidean = np.hstack((prev_euclidean, np.atleast_2d(np.array(new_euclidean_list)).T))

    np.save(arr_dir_name + "/euclidean_" + model_type + ".npy", new_euclidean)
    np.save(arr_dir_name + "/freq.npy", new_freq)
    return new_freq, new_euclidean


def get_euclidean(prev_model, new_model, word):
    """
    Compute Euclidean Distance between two vectors
    :param prev_model: Previous embeddings space
    :param new_model: Current embeddings space
    :param word: selected word
    :return: float euclidean distance
    """
    a = new_model[word]
    b = prev_model[word]
    euclidean = norm(a-b)
    return euclidean


def get_corr(t1, t2):
    """
    Compute Spearman Correlation between two time series
    :param t1, t2: time series
    :return: Spearman Correlation value
    """
    spearman_value = stats.spearmanr(t1, t2)[0]
    return spearman_value


def save_alerts(word, correlation, file ):
    #print("Alert raised for word %s with correlation: %.2f"%(word, correlation))
    file.write("%s : %.4f"%(word, correlation))
    file.write("\n")


def get_threshold_value(corr):
    """
    Estimate threshold value depending on estimated mean and variance
    Threshold should be at 97.5th quantile of Gaussian
    :param corr: correlation tab with last column equal new
    :return: float threshold value
    """
    corr = corr.astype(float)
    corr = np.nan_to_num(corr, copy=True, nan=0.0)

    n_V = corr.shape[0]

    # Transforming to Gaussian z-score
    z_corr = np.log((1+corr)/(1-corr))/2
    z_corr = np.nan_to_num(z_corr)

    new_z_ro = np.mean(z_corr[:, :-1])
    print(new_z_ro.shape)
    curr_corr = z_corr[:, -1]
    print(curr_corr.shape)

    se = np.sqrt(np.sum(np.power(curr_corr - new_z_ro, 2))/(n_V - 1))

    n = (curr_corr - new_z_ro)/se

    threshold = np.tanh(-1.96 * se + new_z_ro)
    print(threshold)

    return threshold


def update_correlation(freq, euclidean, size, arr_dir_name, vocab,date, model_type):
    """
    Update correlation array and send alerts if it cross a threshold
    :param freq: entire frequency array
    :param model_type: SVD or SGNS
    :param euclidean: entire frequency euclidean
    :param size: window size for correlation
    :param arr_dir_name: Directory to fetch correlation array
    :return: updated correlation array
    """
    prev_corr = np.load(arr_dir_name + "/correlation_" + str(model_type) + ".npy")
    f = open(arr_dir_name + "/alerts_" + str(model_type) + ".txt", "a")
    f.write(str(date))
    f.write("\n")
    small_freq = freq[:, -size:]  # Get n last elements of freq
    small_euclidean = euclidean[:, -size:]  # Get n last elements of euclidean

    new_correlation_list = list()
    for i in range(prev_corr.shape[0]):  # For each entry (line) in corr
        new_correlation = get_corr(small_euclidean[i], small_freq[i])
        new_correlation_list.append(new_correlation)
    new_correlation = np.hstack((prev_corr, np.atleast_2d(np.array(new_correlation_list)).T))

    threshold = get_threshold_value(new_correlation)

    for i in range(len(new_correlation_list)):
        if new_correlation_list[i] < threshold:
            save_alerts(vocab[i], new_correlation_list[i], f)

    np.save(arr_dir_name + "/correlation_" + str(model_type) + ".npy", new_correlation)
    return new_correlation


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--category', metavar='category', type=str, help="Category to use as emerging")
    p.add_argument('--model', metavar='model', type=str, help="Either SVD or SGNS for modelization")

    args = p.parse_args()
    cat = args.category
    model_type = args.model

    if cat:
        evaluation_mode = True
        name = "scenarios/emergent_" + str(cat) + ".csv"
        dir_name = "models/" + str(cat).replace(" ", "")
        arr_dir_name = "arrays/" + str(cat).replace(" ", "")
        df = get_data(name)
    else:
        evaluation_mode = False
        name = "data/nyt.csv"
        df = get_data(name)

    end_date = get_end_date(df) # Stop condition for updating
    end = False
    LANGUAGE = "english"
    size_window_corr = 10
    size_init = 12

    while not end:
        if already_exist(dir_name, model_type):
            if model_type == "SGNS":
                prev_model, last_date = get_last_model(dir_name,model_type)  # Previous model and previous date
                prev_ucounts = None
                prev_sgcounts = None
                toks = None
            elif model_type == "SVD":
                prev_model, last_date, prev_ucounts, prev_sgcounts, prev_wcounts, toks = get_last_model(dir_name, model_type)  # Previous model and previous date and prev unigram counts and prev skipgram counts

            df_new, new_date = fetch_new_data(df, last_date, size_init)  # New data and new date
            # Updating model and frequency array
            new_model, frequency, vocab = update_modeling(df_new, dir_name, new_date, prev_model,model_type, prev_ucounts, prev_sgcounts, toks)

            before_model = get_previous_model(dir_name,model_type)

            # Get movement and saving it in a numpy array
            new_freq, new_euclidean = update_movement(arr_dir_name, frequency, vocab, before_model, new_model,model_type)

            if new_freq.shape[1] > size_window_corr and new_euclidean.shape[1] > size_window_corr:  # Check if enough value to compute correlation
                new_corr = update_correlation(new_freq, new_euclidean, size_window_corr, arr_dir_name, vocab, new_date, model_type)

            if new_date == end_date:
                end = True
        else:
            list_date_init = list(df.mois.unique())[:size_init]
            print("Month to considerate for init: ")
            print(list_date_init)
            df_init = df[df.mois.isin(list_date_init)]
            print("Nb of docs for init: %d"%len(df_init))
            model, vocab, frequency = initialize_model(df_init, dir_name, arr_dir_name, model_type, is_init=True)  # Getting init.model and list of vocab
            initialize_arrays(vocab, frequency, arr_dir_name)  # Saving Arrays of freq, movement and correlation

