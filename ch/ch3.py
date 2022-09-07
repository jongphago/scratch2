import numpy as np
from common.layers import MatMul
from common.util import *
from dataset import ptb


# =======================
#  3.1 추론 기반 기법과 신경망
# =======================


def ex313():
    """
    3.1.3 신경망에서의 단어 처리
    """
    c = np.array([1, 0, 0, 0, 0, 0, 0])
    W = np.random.randn(7, 3)
    h = np.matmul(c, W)
    print(h)
    pass


# ===================
#  3.2 단순한 word2vec
# ===================


def ex321():
    """
    3.2.1 CBOW 모델의 추론 처리
        - 입력: 맥락, N개의 단어
            - 몇개(N)의 맥락을 고려
        - 은닉층: 단어의 분산 표현
            - 은닉층은 N개의 입력 결과의 평균
        - 출력: 단어의 발생 확률
            - one-hot vector
    """
    c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
    c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

    W_in = np.random.randn(7, 3)
    W_out = np.random.randn(3, 7)

    in_layer0 = MatMul(W_in)
    in_layer1 = MatMul(W_in)
    out_layer = MatMul(W_out)

    h0 = in_layer0.forward(c0)
    h1 = in_layer1.forward(c1)
    h = 0.5 * (h0 + h1)
    s = out_layer.forward(h)

    print(s)


# ==================
#  3.3 학습 데이터 준비
# ==================


def ex332():
    text = "You say goodbye and I say hello."
    corpus, word_to_id, id_to_word = preprocess(text)
    contexts, target = create_contexts_target(corpus)
    print(target.shape)
    print(contexts.shape)
    vocab_size = len(word_to_id)
    target = convert_one_hot(target, vocab_size)
    contexts = convert_one_hot(contexts, vocab_size)
    print(target.shape)
    print(contexts.shape)


# ==================
#  3.4 CBOW 모델 구현
# ==================

