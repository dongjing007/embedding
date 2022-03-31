import os, tempfile
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.mllib.feature import Word2Vec, Word2VecModel
from pyspark.ml.linalg import Vectors
import random
from collections import defaultdict
import numpy as np
from pyspark.sql import functions as F

class UdfFunction:
    @staticmethod
    def sortF(post_list, timestamp_list):
        """
        sort by time and return the corresponding movie sequence
        eg:
            input: movie_list:[1,2,3]
                   timestamp_list:[1112486027,1212546032,1012486033]
            return [3,1,2]
        """
        pairs = []
        # 组成新数据list
        for m, t in zip(post_list, timestamp_list):
            pairs.append((m, t))
        # 基于每个数据体第二项排序，默认由低到高
        pairs = sorted(pairs, key=lambda x: x[1])
        # 返回每个数据体首元素，实现按时间序列生成用户喜好集
        return [x[0] for x in pairs]

def processItemSequence(spark, rawSampleDataPath):
    # 导入数据
    ratingSamples = spark.read.format("csv").option("header", "true").load(rawSampleDataPath)
    # 预处理数据
    sortUdf = udf(UdfFunction.sortF, ArrayType(StringType()))
    # agg聚合函数 .where(F.col("score") >= 0) \
    userSeq = ratingSamples \
        .groupBy("userId") \
        .agg(sortUdf(F.collect_list("postId"), F.collect_list("posttime")).alias('post_ids')) \
        .withColumn("post_id_str", array_join(F.col("post_ids"), " "))
    # 操作moviceIdStr组成的数据集，串联起来生成string返回
    return userSeq.select('post_id_str').rdd.map(lambda x: x[0].split(' '))

def trainItem2vec(spark, samples, embLength, embOutputPath, saveToRedis, redisKeyPrefix):
    # 设定word2vec参数，embedding值长度、设置窗口大小、设置迭代次数
    word2vec = Word2Vec().setVectorSize(embLength).setWindowSize(5).setNumIterations(10)
    # 模型训练
    model = word2vec.fit(samples)
    # 定义输出地址
    embOutputDir = '/'.join(embOutputPath.split('/')[:-1])
    if not os.path.exists(embOutputDir):
        os.makedirs(embOutputDir)
    # 输出模型结果到execl中
    with open(embOutputPath, 'w') as f:
        for post_id in model.getVectors():
            vectors = " ".join([str(emb)
                               for emb in model.getVectors()[post_id]])
            f.write(post_id + ":" + vectors + "\n")
    return model

def similarRecommended(spark, rawSampleDataPath, model, embLength, embOutputPath, saveToRedis, redisKeyPrefix):
    # 生成spark数据对象
    ratingSamples = spark.read.format("csv").option("header", "true").load(rawSampleDataPath)
    # 通过post_id
    result = ratingSamples.select('userId,postId').collect()
    with open(embOutputPath, 'w') as f:
        for row in result:
            res = []
            res1 = ''
            try:
                # 获取相似的五十笔物件
                syms = model.findSynonyms(row[0], 50)
            except:
                continue
            else:
                for synonym, cosineSimilarity in syms:
                    res.append(str(synonym))
                res1 = "_".join(res)
                f.write(row[0] + "," + (res1) +  "\n")

if __name__ == '__main__':
    conf = SparkConf().setAppName('ctrModel').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    # 数据源地址（请用本地地址替换）
    file_path = '/Users/dj/Desktop/embedding'
    rawSampleDataPath = "test.csv"
    # embedding长度设定
    embLength = 10
    # 生成预处理数据
    samples = processItemSequence(spark, rawSampleDataPath)
    # 跑物品特征item2vec模型
    model = trainItem2vec(spark, samples, embLength,
                          embOutputPath=file_path + "/item2vecEmbTest.csv", saveToRedis=False,
                          redisKeyPrefix="i2vEmb")
    # 保存item2vec模型
    model.save(spark.sparkContext, file_path+"/test_post_embedding")

    # 加载embedding模型
    word2vec = Word2Vec().setVectorSize(embLength).setWindowSize(5).setNumIterations(10)
    model = Word2VecModel.load(
        spark.sparkContext, file_path + "/test_post_embedding")
            
    # 生成物件相似集合
    similarRecommended(spark, rawSampleDataPath, model, embLength,
                       embOutputPath=file_path + "/workSimilarRecTest.csv", saveToRedis=False, redisKeyPrefix="uEmb")