# import pandas as pd
import numpy as np
import scipy as sp
import operator
from collections import defaultdict
from sklearn.metrics import f1_score

from collections import Counter
from scipy.sparse import csr_matrix

def count_words(corpus):
    cnt = 0
    for doc in corpus:
        cnt += len(doc)
    return cnt
    
    
def remove_stopwords(docs, minlen):
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    return [ [x for x in d if x not in stop_words and len(x) >= minlen] for d in docs ]


def to_lowercase(docs):
    return [ [str.lower(x) for x in d] for d in docs ]


def build_matrix(docs):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat, idx


def build_test_matrix(Xtest, idx):
    r""" Build sparse matrix from a list of documents,
    each of which is a list of word/terms in the document.
    """
    nrows = len(Xtest)
    nnz = 0
    for d in Xtest:
        nnz += len(set(d))
    ncols = len(idx)

    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in Xtest:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            temp = idx.get(k,-1)
            if temp != -1:
                ind[j+n] = temp
                val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1

    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()

    return mat


def csr_info(mat, name="", non_empy=False):
    r""" Print out info about this CSR matrix. If non_empy, 
    report number of non-empty rows and cols as well
    """
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
                name, mat.shape[0], 
                sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 for i in range(mat.shape[0])), 
                mat.shape[1], len(np.unique(mat.indices)), 
                len(mat.data)))
    else:
        print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
                mat.shape[0], mat.shape[1], len(mat.data)) )


# scale matrix and normalize its rows
def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf. 
    Returns scaling factors as dict. If copy is True, 
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else mat


def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat


# ### Cosine simillarity

# In[15]:



# cos(a, b) = a (*) b / |a||b|
#  cos(0th, 1st)

def cos_similarity(matrix1, matrix2):  # returns cosine(i, j)
    dot_product = matrix1.dot(matrix2.T) #.to_dense().item()  # the dot-product between the sparse vectors in mat2
    return dot_product


class k_nearest:
    def __init__(self, K_value, eps=0):
        self.knn_list = list()  # [(simillarity_val, ind)]
        self.k_num = K_value
        self.min_similarity = eps

    def put(self, similarity_mat):
        # saves (similarity_val, test_index, train_index) tuples to arr
        sim_arr = []
        for s in similarity_mat:
            sim_arr.append(list(zip(s.indices, s.data)))
        # sort all vectors by similarity
        sorted_arr = []
        for i in sim_arr:
            i.sort(reverse=True, key=lambda x: x[1])
            sorted_arr.append(i)

        # populate the knn_list
        for sim_i in sorted_arr:
            sim_wrt_j = []
            for counter, sim_val in enumerate(sim_i):
                if sim_val[1] > self.min_similarity:
                    sim_wrt_j.append(sim_val)
                if (counter >= self.k_num - 1) and (len(sim_wrt_j) == 0):  # at least 1 elem
                    sim_wrt_j.append(sim_i[0])
                    break
            self.knn_list.append(sim_wrt_j)


ctr_result = list()


def knn_clasifier(X_train: sp.sparse.csr.csr_matrix, y_train: list, X_test: sp.sparse.csr.csr_matrix, similarity_fn,
                  K_value=1, epsilon=0):
    # set up memory for results
    # csr.data , csr.indices, csr.indptr
    predictions = np.zeros(X_test.shape[0], dtype=np.str)
    '''
    #Mothod to print individual results. location sensitive.
    def print_stats(y_test_base_index, pred_val, i):
        pred = int(pred_val)
        Y_label = int(labels[y_test_base_index+i])
        ctr_result.append(pred == Y_label)
        print("pred: ", pred, " Actual: ", Y_label, " - ", pred == Y_label)

    '''
    for i in range(X_test.shape[0]):
        knn = k_nearest(K_value, epsilon)

        sim_mat = similarity_fn(X_test[i], X_train)

        knn.put(sim_mat)

        val = list()
        for l in knn.knn_list:
            freq_dict = defaultdict(float)
            for ind in l:
                freq_dict[ind[0]] += ind[1]
            val.append(freq_dict)

        # sort val list of dicts by (test)indexes to get class label
        ind_val = [sorted(t.items(), key=lambda x: x[1], reverse=True)[0][0] for t in val]
        pred_list = [y_train[i] for i in ind_val]  # pred_list contains only 1 element

        for p in pred_list:
            predictions[i] = pred_list[0]

        print(i)
    return predictions




'''
def cos_similarity(matrix1, matrix2, i, j):  # returns cosine(i, j)
    dot_product = matrix1[i].dot(matrix2[j].T).todense().item()  # the dot-product between the sparse vectors in mat2
    cos = dot_product / (norm(matrix1[i]) * norm(matrix2[j]))
    return cos


# ## kNN clasifier  ##################################
class k_nearest:
    def __init__(self, K_value, eps=0):
        self.arr = list() # [(simillarity_val, ind)]
        self.k_num = K_value
        self.min_similarity = eps
    
    def put(self, test_index, train_index, similarity_value): 
        # saves (similarity_val, test_index, train_index) tuples to arr
        self.arr.append((similarity_value, test_index, train_index))
        self.arr.sort(reverse=True, key=lambda x: x[0])
#         if len(self.arr) >= 1 and similarity_value < self.min_similarity: # epsilon cond.
#             del self.arr[-1:]
#             return
        lowest_sim = self.arr[-1][0]
        epsilon_cond = len(self.arr) > 1 and (lowest_sim < self.min_similarity)        
        if(len(self.arr) > self.k_num or epsilon_cond): # consider only k nearest
            del self.arr[-1:]


ctr_result = list()
# ratio_20 = int(mat_norm.shape[0] * 0.20)

def knn_clasifier(X_train: sp.sparse.csr.csr_matrix, y_train: list, X_test: sp.sparse.csr.csr_matrix, similarity_fn, K_value=1, epsilon=0.0):
    # set up memory for results
    # csr.data , csr.indices, csr.indptr
    predictions = np.zeros(X_test.shape[0], dtype=np.str)
    
    #Mothod to print individual results. location sensitive.
    def print_stats(y_test_base_index, pred_val, i):
        pred = int(pred_val)
        Y_label = int(labels[y_test_base_index+i])
        ctr_result.append(pred == Y_label)
        print("pred: ", pred, " Actual: ", Y_label, " - ", pred == Y_label)
        

    for i in range(X_test.shape[0]):
        knn = k_nearest(K_value, epsilon)
        for j in range(X_train.shape[0]):
            sim = similarity_fn(X_test, X_train, i, j)
            knn.put(i, j, sim)           
        
        nn_labels_freq = dict()
        # majority voting with weight
        for n in knn.arr:
            if(y_train[n[2]] in nn_labels_freq.keys()):
                nn_labels_freq[y_train[n[2]]] += (1*n[0])
            else:
                nn_labels_freq[y_train[n[2]]] = (1*n[0])
        # mul weights
#         for n in knn.arr:
#             nn_labels_freq[y_train[n[2]]]
        
        max_freq_class = sorted(nn_labels_freq.items(), reverse=True, key=operator.itemgetter(1))
        predictions[i] = max_freq_class[0][0]
        # print("-for doc", i, " | knn.sim: ", [(m1_y[sim[2]], sim[0]) for sim in knn.arr[:]]) # [sim[0] for sim in knn.arr[:]]
        print_stats(mat_norm.shape[0]-2887, predictions[i], i)   # print individual results: change 1st param as needed
    return predictions
'''


def pre_process(docs):
    docs_new = remove_stopwords(docs, 4)
    # print("total words in entire docs set:", count_words(docs))
    # print("After removing stopwords:", count_words(docs_new))
    docs_new = to_lowercase(docs_new)
    return docs_new


def idf_norm(csr_matrix):
    csr_info(csr_matrix)

    # Scale values by multiplying corresponding idf values.
    # Then, normalize it by L2 norm
    mat_idf = csr_idf(csr_matrix, copy=True)
    mat_norm = csr_l2normalize(mat_idf, copy=True)

    # csr_info(mat_norm)
    return mat_norm


def write_results(file_path, result):
    with open(file_path, 'w') as f:
        for i in range(len(result)):
            f.write(result[i]+'\n')



if __name__ == '__main__':

    # open docs file and read its lines
    with open("train.dat", "r") as fh:
        lines = fh.readlines()

    # split input into labels and docs
    labels = []
    docs = []
    for l in lines:
        inp_data = l.split('\t')
        labels.append(inp_data[0])
        docs.append(inp_data[1].split())  # list of list of words

    # test.dat file
    with open("test.dat", "r") as fh:
        test_lines = fh.readlines()
    test_docs = []
    for l in test_lines:
        test_docs.append(l.split())  # list of list of words


    #### pre-processing
    print("Pre-processing ....")
    doc_train = pre_process(docs)
    doc_test = pre_process(test_docs)

    train_mat, idx = build_matrix(doc_train)
    test_mat = build_test_matrix(doc_test, idx)
    train_mat = idf_norm(train_mat)
    test_mat = idf_norm(test_mat)

    print("Pre-processing done.")

    print("\nkNN Start....")

    #### Apply kNN
    # ratio_20 = int(mat_norm.shape[0] * 0.20) # = 2887

    m1_X = train_mat
    m1_y = labels[:]
    t1_X = test_mat

    t1_p = knn_clasifier(X_train=m1_X, y_train=m1_y, X_test=t1_X, similarity_fn=cos_similarity, K_value=10, epsilon=0.6)

    print('*' * 20)

    print("\nkNN done.")

    # for i in range(len(t1_p)):
    #     print("pred: ", t1_p[i], " Actual: ", t1_y[i], " - ", int(t1_p[i]) == int(t1_y[i]))

    # print('*' * 20)
    # cntr = Counter(m1_y)
    # print("label frequencies: ", cntr.most_common())
    # cnt_results = Counter(ctr_result)
    # print("Result stats: ", cnt_results.most_common())


    # ### F1-scoring
    # accuracy = f1_score(t1_y, t1_p, average='micro')
    # print("F1-score: ", accuracy)

    print("\nWriting results to file.")
    write_results('out.txt', t1_p)

