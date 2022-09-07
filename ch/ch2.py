import matplotlib.pyplot as plt

from common import *
from dataset import ptb
np.set_printoptions(precision=3)


text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)


# =================
#  2.3 통계 기반 기법
# =================


def ex231():
    """
    2.3.1 파이썬으로 말뭉치 전처리하기
        - implemet function:
            - preprocess()
    """
    corpus, word_to_id, id_to_word = preprocess(text)
    print(corpus)
    print(word_to_id)
    print(id_to_word)


def ex232():
    """
    2.3.2 단어의 분산 표현
        - 단어를 RGB 처럼 표현
        - 분산 표현 distributional representation
            - fixed length dense vector
    """
    pass


def ex233():
    """
    2.3.3 분포 가설
        - '단어의 의미는 주변 단어에 의해 형성된다'
            - 단어는 의미가 없다
            - 단어가 사용된 '맥락(context)'이 의미를 형성한다
        - "You say goodbye and i say hello."
            - 윈도우 크기, window size
                - 좌우
    """
    pass


def ex234():
    """
    2.3.4 동시발생 행렬, co-occurrence matrix
        - 주변 단어를 세어 보는 방법
            - 주변에 어떤 단어가 몇 번이나 등장하는지 집계
            - 통계 기반 기법
        - function implement: creat_co_matrix
    """
    co_matrix = create_co_matrix(corpus, vocab_size)
    print(co_matrix)


def ex235():
    """
    2.3.5 벡터 간 유사도
        - 코사인 유사도, cosine similarity
    """
    c0 = C[word_to_id['you']]
    c1 = C[word_to_id['i']]
    print(cos_similarity(c0, c1))


def ex236():
    """
    2.3.6 유사 단어의 랭킹 표시
        - function implementation
            - most_similar
    """
    query = 'you'
    most_similar(query, word_to_id, id_to_word, C)

# ========================
#  2.4 통계 기반 기법 개선하기
# ========================


def ex241():
    """
    2.4.1 상호정보량
        - [problem] 동시 발생 빈도를 기준으로 비교하면 'car'는 'drive'보다 'the'와 유사도가 높다.
            - 'the', 'car' & 'car', 'drive
        - 점별 상호정보량, Pointwise Mutual Information
            - P(x, y) / (P(x) * P(y))
            - 각 단어의 발생빈도를 정규화
                - 모든 단어가 동일한 횟수로 발생 하였을때 동시 발생 횟수를 비교
                - PPMI: Positive PMI
            - cons
                - 원소 대부분이 0
                - 노이즈에 약하고 견고하지 못하다
            - 차원 감소(2.4.2)로 해결
    """
    W = ppmi(C)
    print(W)
    pass


def ex242():
    """
    2.4.2 차원 감소, dimensionalitiy reduction

    """


def ex243():
    """
    2.4.3 SVD에 의한 차원 감소
        - 특잇값분해 Singular Value Decomposition
                - X = USV^T
                    - U : decreased word space
                    - S : singular value
    """
    W = ppmi(C)
    U, S, V = svd(W)
    print(U.shape)
    print(S.shape)
    print(S)
    print(V.shape)

    for word, word_id in word_to_id.items():
        plt.annotate(word, U[word_id, 0:2])
    plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
    plt.show()
    pass


def ex244():
    """
    2.4.4 PTB 데이터셋
    """
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    print(len(corpus))
    pass


def ex245():
    """
    2.4.5 PTB 데이터셋 평가
    """
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)

    # 동시 발생 행렬
    C = create_co_matrix(corpus, vocab_size)

    # PPMI
    W = ppmi(C, verbose=True)

    # SVD
    wordvec_size = 100
    U, S, V = svd(W, wordvec_size=wordvec_size)

    word_vecs = U[:, :wordvec_size]

    querys = ['you', 'year', 'car', 'hyundai']
    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs)


    pass
