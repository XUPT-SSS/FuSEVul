import pandas as pd
import re
from clean_gadget import clean_gadget
from lang_processors.cpp_processor import CppProcessor
import csv
import time
import numpy as np

def normalization(source):
    """
    normalization code
    :param source: dataframe
    :return:
    """
    cpp_processor = CppProcessor()
    nor_code = []
    for fun in source['code']:
        lines = fun.split('\n')
        # print(lines)
        code = ''
        for line in lines:
            line = line.strip()
            line = re.sub('//.*', '', line)
            code += line + ' '
        # code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
        code = re.sub('/\\*.*?\\*/', '', code)
        code = clean_gadget([code])
        code[0] = re.sub('"".*?""', '', code[0], 20)
        code_list = cpp_processor.tokenize_code(code[0])
        print(len(code_list))

        tokenization_code = ''
        for token in code_list:
            tokenization_code = tokenization_code + token + " "
        nor_code.append(tokenization_code)
        # print(tokenization_code)
    return nor_code


def normalization2(source):
    cpp_processor = CppProcessor()
    nor_code = []
    for fun in source['code']:
        lines = fun.split('\n')
        # print(lines)
        code = ''
        for line in lines:
            line = line.strip()
            line = re.sub('//.*', '', line)
            code += line + ' '
        # code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
        code = re.sub('/\\*.*?\\*/', '', code)
        code = clean_gadget([code])
        code[0] = re.sub('"".*?""', '', code[0], 20)

        code_list = cpp_processor.tokenize_code(code[0])
        # nor_code.append(code[0])
        # nor_code.append(code6)
        tokenization_code = ''
        for token in code_list:
            tokenization_code = tokenization_code + token + " "
        nor_code.append(tokenization_code)
        # print(tokenization_code)
        with open('./corpus.txt', 'a') as f:
            f.write(tokenization_code)
            f.write('\n')
    return nor_code


def mutrvd():
    train = pd.read_pickle('trvd_train.pkl')
    test = pd.read_pickle('trvd_test.pkl')
    val = pd.read_pickle('trvd_val.pkl')

    train['code'] = normalization(train)
    train.to_pickle('./mutrvd/train.pkl')

    test['code'] = normalization(test)
    test.to_pickle('./mutrvd/test.pkl')

    val['code'] = normalization(val)
    val.to_pickle('./mutrvd/val.pkl')


def nor(source):
    cpp_processor = CppProcessor()
    nor_code = []
    lines = source.split('\n')
    # print(lines)
    code = ''
    for line in lines:
        line = line.strip()
        line = re.sub('//.*', '', line)
        code += line + ' '
    # code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
    code = re.sub('/\\*.*?\\*/', '', code)
    code = clean_gadget([code])
    # code[0] = code[0].replace('"".*?""', '', 10)
    code[0] = re.sub('"".*?""', '', code[0], 20)

    code_list = cpp_processor.tokenize_code(code[0])
    tokenization_code = ''
    for token in code_list:
        tokenization_code = tokenization_code + token + " "
    nor_code.append(tokenization_code)
    # print(tokenization_code)
    return nor_code


if __name__ == '__main__':
    # str = r"""
    # void PEM_dek_info(char *buf, const char *type, int len, char *str)
    # {static const unsigned char map[17] = ""0123456789ABCDEF"";
    # long i;int j;
    # OPENSSL_strlcat(buf, ""DEK-Info: "", PEM_BUFSIZE);
    # OPENSSL_strlcat(buf, type, PEM_BUFSIZE);
    # OPENSSL_strlcat(buf, "","", PEM_BUFSIZE);
    # j = strlen(buf);
    # if (j + (len * 2) + 1 > PEM_BUFSIZE)return;
    # for (i = 0; i < len; i++) {
    #     buf[j + i * 2] = map[(str[i] >> 4) & 0x0f];
    #     buf[j + i * 2 + 1] = map[(str[i]) & 0x0f];}
    # buf[j + i * 2] = '\n';
    # buf[j + i * 2 + 1] = '\0';}
    # """

    # str= r'''
    # static av_cold int flic_decode_end ( AVCodecContext * avctx ) { FlicDecodeContext * s = avctx -> priv_data ; if ( s -> frame . data [ 0 ] ) avctx -> release_buffer ( avctx , & s -> frame ) ; return 0 ; }
    # '''
    # code = nor(str)
    # print(code)

    #按函数处理
    start_time = time.time()
    code_train = pd.read_csv('../pytorch-image-models/trvd/test.csv')
    data = pd.DataFrame(code_train)
    list = []
    for i in range(len(data)):
        s = data['text'][i]
        t = nor(s)
        t=' '.join(t)
        list.append(t)
    rows = zip(list)
    with open('../pytorch-image-models/trvd/test_line.csv', "a") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    end_time = time.time()
    execution_time = end_time - start_time
    print("time:", execution_time, "seconds")


    # #按行处理
    # start_time = time.time()
    # code_train = pd.read_csv('../pytorch-image-models/ours/our_train1.csv')
    # data = pd.DataFrame(code_train)
    # print(data)
    # list = []
    # for i in range(len(data)):
    #     s = data['code'][i]
    #     t=''
    #     str = nor(s)[0]
    #     # print(str)
    #     for j in range(len(str)):
    #         if str[j]==';'or str[j]=='{':
    #             t+=str[j]
    #             t+='\n'
    #         else:
    #             t+=str[j]
    #     list.append(t)
    # rows = zip(list)
    # with open('../pytorch-image-models/ours/our_train1_time.csv', "w") as f:
    #     writer = csv.writer(f)
    #     for row in rows:
    #         writer.writerow(row)
    #
    # end_time = time.time()
    # execution_time = end_time - start_time
    #
    # print("time:", execution_time, "seconds")

