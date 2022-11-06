from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from gensim.utils import tokenize
import konlpy
#from eunjeon import Mecab
from pandas import Series, DataFrame
import pandas as pd
import re
import pickle
import re
from konlpy.tag import Okt, Mecab
from sqlalchemy import false, true
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def change_word(sentence):
    df = pd.DataFrame({'comment': sentence}, index=[0])
    mecab = Mecab()
    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘',
                 '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    stopsentence = df['comment'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣]", "", regex=True)
    print(stopsentence)
    tokenized_data = stopsentence.apply(mecab.morphs)
    tokenized_data = tokenized_data.apply(
        lambda x: [item for item in x if item not in stopwords])
    return tokenized_data


def change_word2(s):
    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍',
                 '과', '도', '를', '을', '으로', '자', '에', '와', '한', '하다', '랑']

    okt = Okt()

    s = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", s)
    s = re.sub('^ +', "", s)

    s = okt.morphs(s, stem=True)  # 토큰화
    s = [word for word in s if not word in stopwords]  # 불용어 제거
    return s

# def change_word(sentence):
#     df = pd.DataFrame({'comment': sentence}, index=[0])
#     okt = konlpy.tag.Okt()
#     stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘',
#                  '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
#     sentence = df['comment'].str.replace("[^ㄱ-ㅎㅏ-ㅣ0-9가-힣]", "", regex=True)
#     print(sentence[0])
#     tokenized_sentence = okt.morphs(sentence[0], stem=True)
#     print(tokenized_sentence)
#     mecab=Mecab()

#     #stopwords_removed_sentence=[word for word in tokenized_sentence if not word in stopwords]
#     return tokenized_sentence
#     # return stopwords_removed_sentence


def find(sentence):
    #model_fname = 'C:\\Users\\tpgus\\fasttext_api\\data\\mecab\\fasttext_mecab'
    model_fname = './data/mecab/fasttext_mecab'
    model = FastText.load(model_fname)
    model_dict = {}

    pre_stn = change_word2(sentence)
    print(pre_stn)
    #temp_list = pre_stn.tolist()

    forbidden_words = ['시발', '씨발', 'ㅅㅂ', '십새끼', '개새', '슼갈', '미친놈',
                       '새끼', '개새끼', '씹', '씹련', '시발련', '니애미', 'ㅅㄲ', '지랄', '병신', '븅신', '븅신년',
                       '좆', '애미']
    li = []

    for i in pre_stn:
        # print(i)
        if i in forbidden_words:
            li.append([i])
            continue
        dict = model.wv.most_similar(i)
        for dic in dict:
            print(i)
            print(dic)
            if dic[0] in forbidden_words:
                li.append([i])
                break

    # for i in pre_stn:
    #     for j in i:
    #         # print(j)
    #         if j in forbidden_words:
    #             li.append([j])
    #             break
    #         dict = model.wv.most_similar(j)

    #         for dic in dict:
    #             print(dic)
    #             if dic[0] in forbidden_words:
    #                 li.append([i])
    #                 break

    print(li)
    if len(li) == 0:
        return None
    else:
        model_dict['banned_word'] = li
        return model_dict

        # return li
#find('야이 개새끼야')
#find("슼갈이 왜화남 사실상 샌박이 준우승이라던 슼까들 다 아가리닫게생겼구만ㅋㅋ")


def sentiment_predict(s):
    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍',
                 '과', '도', '를', '을', '으로', '자', '에', '와', '한', '하다', '랑']

    okt = Okt()

    max_len = 30
    #loaded_model = load_model('C:/Users/tpgus/fasttext_api/data/mecab/cnn_model2')
    loaded_model = load_model('./data/mecab/cnn_model2')

    with open('./data/mecab/tokenizer2.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    s = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", s)
    s = re.sub('^ +', "", s)

    s = okt.morphs(s, stem=True)  # 토큰화
    s = [word for word in s if not word in stopwords]  # 불용어 제거

    # tokenizer = Tokenizer() --> wrong
    # tokenizer.fit_on_texts(s) --> wrong
    # print(tokenizer.word_index) --> wrong

    encoded = tokenizer.texts_to_sequences([s])  # 정수 인코딩
    padded = pad_sequences(encoded, maxlen=max_len)  # 패딩
    score = float(loaded_model.predict(padded))

    if (score > 0.5):
        print("{:.2f}% 확률로 긍정입니다.\n".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정입니다.\n".format((1-score) * 100))

    return score
