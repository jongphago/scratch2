from typing import List
Vector = List[float]
import numpy as np
np.set_printoptions(precision=3)


def preprocess(text: str):
    """
    :param text:
    :return: (corpus: np.ndarray, word_to_id: dict, id_to_word: dict)
    """
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus: List, vocab_size: int, window_size: int = 1) -> np.ndarray:
    """
    :param corpus:
    :param vocab_size:
    :param window_size:
    :return: co_matrix
    """
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - i
            right_idx = idx + i
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix


def cos_similarity(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    :param x:
    :param y:
    :param eps:
    :return:
    """
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


def most_similar(query: str,
                 word_to_id: dict[str, int],
                 id_to_word: dict[int, str],
                 word_matrix: np.ndarray,
                 top: int = 5) -> None:
    if query not in word_to_id:
        raise ValueError
    print(f'\n[query] {query}')
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(word_to_id)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(query_vec, word_matrix[i])
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(f' {id_to_word[i]}: {similarity[i]}')
        count += 1
        if count >= top:
            return None


def ppmi(C: np.ndarray, verbose: bool = False, eps: float = 1e-8) -> np.ndarray:
    """
    Convert CO-Occurence matrix to positive pointwise mutual information
    :param C:
    :param verbose:
    :param eps:
    :return:
    """
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    def pmi(i, j):
        return np.log2(C[i, j] * N / (S[j] * S[i]) + eps)

    for row in range(C.shape[0]):
        for col in range(C.shape[1]):
            M[row, col] = max(0, pmi(row, col))
            if verbose:
                cnt += 1
                print(f'{100 * cnt / total:4.1f}% complete')
    return M


def svd(W: np.ndarray, wordvec_size=100) -> tuple:
    """
    Convert PPMI matrix to decreased word space vector
    :param W: PPMI of corpus
    :param wordvec_size:
    :return:
        U: decreased word space
        S: singular value
        _: -
    """
    try:
        from sklearn.utils.extmath import randomized_svd
        U, S, _ = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
    except ImportError:
        U, S, _ = np.linalg.svd(W)
    return U, S, _


def create_contexts_target(corpus: List[int], window_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the context index based on the target index
    :param corpus:
    :param window_size:
    :return: contexts: np.ndarray, target: np.ndarray
    """
    target = corpus[window_size: -window_size]
    contexts = []
    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)
    return np.array(contexts), np.array(target)


def convert_one_hot(corpus: np.ndarray, vocab_size: int) -> np.ndarray:
    N = corpus.shape[0]
    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1
    return one_hot
