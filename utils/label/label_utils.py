#!/usr/bin/env python
from Levenshtein import *
import numpy as np
import logging
import re
import os

logger = logging.getLogger("Data_Util")

rex = re.compile(' ')


def caculate_edit_distance(preds, labels):
    distances = [distance(p, l) for p, l in zip(preds, labels)]
    return sum(distances) / len(distances)


# pred[seq,3770] => xxxx
def prob2str(pred, charset):
    # 得到当前时间的输出，是一个3770的概率分布，所以要argmax，得到一个id
    decoder_index = np.argmax(pred, axis=-1)  # decoder_index[seq]
    # logger.debug("decoder_index:%r",decoder_index)
    return id2str(decoder_index, charset)


# result[b,seq] => [xx,yy,..,zz]
def ids2str(results, characters):
    values = []
    for r in results:  # 每个句子
        values.append(id2str(r))
    return values


# id[1,3,56,4,35...] => xyzqf...
def id2str(ids, characters):
    str = [characters[int(id)] for id in ids]  # 每个字
    result = ''.join(c for c in str if c != '\n')
    return result


# 'c' => 215
def str2id(str_val, characters):
    if not str_val in characters:
        logger.warning("字符[{}]在字典中不存在".format(str_val))
        return 0
    return characters.index(str_val)


# 'abc' => [213,214,215]
def strs2id(strings, characters):
    ids = []
    for c in strings:
        ids.append(str2id(c,characters))
    return ids


# load charset, the first one is foreground, left are characters
def get_charset(charset_file):
    charset = open(charset_file, 'r', encoding='utf-8').readlines()
    charset = [ch.strip("\n") for ch in charset]
    charset = "".join(charset)
    charset = list(charset)
    charset.insert(0, ' ')  # this is important to for character map
    logger.info(" Load character table, totally [%d] characters", len(charset))
    return charset


# 字符串
def caculate_accuracy(preds, labels):
    result = [p == l for p, l in zip(preds, labels)]
    return np.array(result).mean()


# 从文件中读取样本路径和标签值
# >data/train/21.png )beiji
# >data/train/22.png 市平谷区金海
# >data/train/23.png 江中路53
# bin_num:分箱个数
def read_data_file(label_file_name, process_num=None):
    f = open(label_file_name, 'r', encoding="utf-8")
    data = []
    count = 0
    for line in f:

        if process_num and count > process_num:
            logger.debug("加载完成！仅仅加载[%d]条数据，", process_num)
            break

        filename, _, label = line[:-1].partition(' ')  # partition函数只读取第一次出现的标志，分为左右两个部分,[:-1]去掉回车
        # print(filename,":",label)
        data.append((filename, label))
        count += 1
    f.close()
    return data


def load_labels(label_dir):
    files = os.listdir(label_dir)
    image_labels = []
    for f in files:
        label_path = os.path.join(label_dir, f)
        name, ext = os.path.splitext(f)

        if ext.upper() != ".JSON" and ext.upper() != ".TXT": continue

        image_path = None
        image_subfix = [".jpg", ".png", ".jpeg"]
        for subfix in image_subfix:
            __image_path = os.path.join(label_dir, name + subfix)

            if os.path.exists(__image_path):
                image_path = __image_path
                break

        if image_path:
            image_labels.append([image_path, label_path])

    return image_labels


# !!! 此方法已废弃，加载目前采用fit_generator的multiprocess=True+Work=10的方式，不用自己去创建多进程了
# 从文件中读取样本路径和标签值，并放入分箱中，为了是每个箱子多进程加载
# >data/train/21.png )beiji
# >data/train/22.png 市平谷区金海
# >data/train/23.png 江中路53
# bin_num:分箱个数
def read_data_file_bucket(label_file_name, process_num):
    f = open(label_file_name, 'r', encoding="utf-8")
    data = []
    for line in f:
        filename, _, label = line[:-1].partition(' ')  # partition函数只读取第一次出现的标志，分为左右两个部分,[:-1]去掉回车
        # print(filename,":",label)
        data.append((filename, label))
    f.close()

    logger.debug("从[%s]中读取了所有原始数据，一共[%d]行", label_file_name, len(data))

    # chunks函数用于分箱
    def chunks(l, step):
        for i in range(0, len(l), step):
            yield l[i:i + step]

    # print(len(data),process_num)
    data_list = list(chunks(data, len(data) // process_num))
    # print(data_list)
    logger.debug("所有数据[%d]条，被分箱到[%d]中", len(data), process_num)

    return data_list


def process_lines(charsets, data):
    result = []
    for d in data:
        # print(d)
        file, label = d
        filename, labels_index = process_line(file, label, charsets)
        if filename is None: continue
        if labels_index is None: continue
        result.append((filename, labels_index))
    return result


# 处理每一行数据：data/train/22.png 市平谷区金海
# 返回的是filename,labels_index
def process_line(filename, label, charsets):
    if not os.path.exists(filename):
        logger.warning("标签文件[%s]不存在啊", filename)
        return None, None

    processed_label = process_unknown_charactors(label, charsets)
    if processed_label is None or len(processed_label) == 0:
        logger.error("解析标签字符串失败，忽略此样本：[%s]", label)
        return None, None

    labels_index = convert_labels_to_ids(processed_label, charsets)
    if labels_index is None:
        return None, None

    return filename, labels_index


# labels是所有的标签的数组['我爱北京','我爱天安门',...,'他说的法定']
# characters:词表
def convert_to_id(labels, characters):
    _lables = []
    for one in labels:
        _lables.append([characters.index(l) for l in one])

    return _lables


def process_unknown_charactors_all(all_sentence, dict, replace_char=None):
    result = []
    for sentence in all_sentence:
        result.append(process_unknown_charactors(sentence, dict, replace_char))
    return result


# 1.处理一些“宽”字符,替换成词表里的
# 2.易混淆的词，变成统一的
# 3.对不认识的词表中的词，是否替换成某个字符，如果不与替换，就直接返回空
def process_unknown_charactors(sentence, dict, replace_char=None):
    unkowns = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ！＠＃＄％＾＆＊（）－＿＋＝｛｝［］｜＼＜＞，．。；：､？／×·■"
    knows = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()-_+={}[]|\<>,.。;:、?/x.."

    result = ""

    # 先去除空格
    sentence = rex.sub('', sentence)

    for one in sentence:
        # 对一些特殊字符进行替换，替换成词表的词
        i = unkowns.find(one)
        if i == -1:
            letter = one
        else:
            letter = knows[i]
            # logger.debug("字符[%s]被替换成[%s]", one, letter)

        # 看是否在字典里，如果不在，给替换成一个怪怪的字符'■'来训练，也就是不认识的字，都当做一类，这个是为了将来识别的时候，都可以明确识别出来我不认识，而且不会浪费不认识的字的样本
        # 但是，转念又一想，这样也不好，容易失去后期用形近字纠错的机会，嗯，算了，还是返回空，抛弃这样的样本把
        if letter not in dict:
            if replace_char:
                letter = replace_char  # '■'
            else:
                logger.error("句子[%s]的字[%s]不属于词表,剔除此样本", sentence, letter)
                return None

        result += letter
    return result


# 将labels转换为one_hot, "我爱北京"=> [(0,0,0,......,0,1,0,.....0,0),(0,0,0,......,0,1,0,.....0,0),..]维度是词表维度
def convert_labels_to_ids(label, charsets):
    labels_index = []
    for l in label:
        if not l in charsets:
            logger.error("字符串[%s]中的字符[%s]未在词表中", label, l)
            return None
        labels_index.append(charsets.index(l))
    return labels_index
