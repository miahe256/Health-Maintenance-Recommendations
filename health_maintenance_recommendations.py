import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import matplotlib.pyplot as plt
import numpy as np
import jieba
import jieba.analyse

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建停用词列表（包含中英文）
STOPWORDS = {
    # 英文停用词
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 
    'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
    'or', 'because', 'as', 'until', 'while',
    
    # 中文停用词
    '的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
    '之', '在', '上', '也', '因此', '但是', '并且', '或者', '不过', '然后',
    '这', '那', '这个', '那个', '这些', '那些', '这样', '那样', '这么', '那么',
    '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
    '一个', '一些', '某', '某个', '某些', '我', '你', '他', '她', '它', '们',
    
    # 数字和单位
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    '个', '次', '种', '项', '条',
    
    # 特殊词
    '推荐', '建议', '可以', '应该', '需要', '如何', '怎么', '一定', '必须',
    '提供', '进行', '采用', '使用', '通过', '以及', '等等', '由于', '因为', '食物','抗氧化','益处',
    '帮助', '促进', '改善', '提高', '增加', '减少', '预防', '讲解', '方法', '富含', '探讨', '分析', '介绍', '作用', '控制'
}

class ChineseTokenizer:
    """自定义中文分词器"""
    def __init__(self):
        # 添加自定义词典（如果有的话）
        # jieba.load_userdict("custom_dict.txt")
        pass

    def __call__(self, text):
        # 使用jieba进行分词
        words = jieba.cut(text)
        # 过滤停用词和空字符
        return [word for word in words if word.strip() and word not in STOPWORDS]

def clean_text(text):
    """
    文本清洗函数
    :param text: 输入文本
    :return: 清洗后的文本
    """
    if not isinstance(text, str):
        return ""
    
    # 预处理特殊短语（在分词前保护某些词组）
    text = text.lower()
    
    # 扩展特殊短语映射
    special_phrases = {
        'omega-3': 'omega3食物',
        'omega3': 'omega3食物',
        'ω-3': 'omega3食物',
        'ω3': 'omega3食物',
        '3型': 'omega3食物',
        'omega-6': 'omega6食物',
        'omega6': 'omega6食物',
        'ω-6': 'omega6食物',
        'ω6': 'omega6食物',
        '6型': 'omega6食物',
        '3的食物': 'omega3食物',
        '3 的食物': 'omega3食物',
        '3的 食物': 'omega3食物',
        '3 的 食物': 'omega3食物'
    }
    
    # 按照短语长度排序，确保先替换最长的短语
    sorted_phrases = sorted(special_phrases.items(), key=lambda x: len(x[0]), reverse=True)
    for phrase, replacement in sorted_phrases:
        text = text.replace(phrase, replacement)
    
    # 移除特殊字符，但保留中文和英文单词
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    
    # 移除独立的数字
    text = re.sub(r'\b\d+\b', '', text)
    
    # 使用jieba分词
    words = jieba.cut(text)
    
    # 过滤停用词并重新组合
    text = ' '.join(word for word in words if word.strip() and word not in STOPWORDS)
    
    return text.strip()

def get_top_n_words(corpus, n=1, k=None):
    """
    获取文本语料库中出现频率最高的n-gram特征
    :param corpus: 文本语料库
    :param n: n-gram中的n
    :param k: 返回前k个高频词
    :return: 返回词频统计结果
    """
    # 打印一些调试信息
    print("\n=== 文本处理调试信息 ===")
    print("处理前的部分示例文本：")
    for text in list(corpus)[:3]:
        print(f"- {text[:100]}...")
    
    # 使用自定义的中文分词器
    tokenizer = ChineseTokenizer()
    
    # 统计ngram词频矩阵
    vec = CountVectorizer(
        tokenizer=tokenizer,
        ngram_range=(n, n),
        min_df=2,  # 至少出现2次的词才会被统计
        max_df=0.95  # 忽略出现在95%以上文档中的词
    )
    
    # 构建词频矩阵
    try:
        vec.fit(corpus)
        bag_of_words = vec.transform(corpus)
        
        # 获取词汇表
        vocabulary = vec.get_feature_names_out()
        
        # 计算词频
        sum_words = bag_of_words.sum(axis=0).A1
        
        # 创建词频对
        words_freq = [(word, freq) for word, freq in zip(vocabulary, sum_words)]
        
        # 按照词频从大到小排序
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        
        # 过滤掉不需要的词
        filtered_words_freq = []
        for word, freq in words_freq:
            # 过滤掉纯数字、单个字符的词
            if not word.isdigit() and len(word) > 1:
                filtered_words_freq.append((word, freq))
        
        # 打印词频统计结果
        print("\n词频统计结果：")
        for word, freq in filtered_words_freq[:20]:
            print(f"- {word}: {freq}")
        
        return filtered_words_freq[:k]
    except Exception as e:
        print(f"词频统计时发生错误: {str(e)}")
        return []

def plot_top_words(words_freq, title='高频词统计'):
    """
    绘制高频词统计图
    :param words_freq: 词频统计结果
    :param title: 图表标题
    """
    # 过滤掉可能的空词
    words_freq = [(word, count) for word, count in words_freq if word.strip()]
    
    df = pd.DataFrame(words_freq, columns=['word', 'count'])
    plt.figure(figsize=(12, 6))
    df.groupby('word').sum()['count'].sort_values().plot(
        kind='barh', 
        title=title
    )
    plt.xlabel('频次')
    plt.ylabel('词语')
    plt.tight_layout()
    plt.show()

def create_health_recommendations(data_df):
    """
    创建健康推荐系统
    :param data_df: 包含健康数据的DataFrame
    :return: 推荐函数
    """
    # 确保数据框包含必要的列
    required_columns = ['文章标题', '中心内容描述']
    if not all(col in data_df.columns for col in required_columns):
        raise ValueError(f"数据框必须包含 '{required_columns[0]}' 和 '{required_columns[1]}' 列")
    
    # 设置索引为文章标题
    if '文章标题' in data_df.columns:
        data_df.set_index('文章标题', inplace=True)
    else:
        raise ValueError("CSV文件中缺少 '文章标题' 列，无法设置为索引。")

    # 清洗文本
    print("\n=== 文本清理过程 ===")
    print("清理前的示例：")
    print(data_df['中心内容描述'].iloc[0])
    
    # 清洗文本
    data_df['clean_中心内容描述'] = data_df['中心内容描述'].apply(clean_text)
    
    print("\n清理后的示例：")
    print(data_df['clean_中心内容描述'].iloc[0])
    print("=== 文本清理结束 ===\n")
    
    # 使用TF-IDF提取文本特征
    tfidf = TfidfVectorizer(
        tokenizer=ChineseTokenizer(),
        ngram_range=(1, 3),
        min_df=0.01,
        max_df=0.95,  # 忽略出现在95%以上文档中的词
    )
    
    # 构建TF-IDF矩阵
    tfidf_matrix = tfidf.fit_transform(data_df['clean_中心内容描述'])
    
    # 计算余弦相似度
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    def get_recommendations(title, n=5):
        """
        基于文章标题获取推荐
        :param title: 文章标题
        :param n: 推荐数量
        :return: 推荐列表 (标题, 分数)
        """
        try:
            # 确保标题存在于索引中
            if title not in data_df.index:
                print(f"错误：找不到标题为 '{title}' 的文章")
                return []
            
            # 获取标题对应的位置
            idx = data_df.index.get_loc(title)
            
            # 获取该位置的相似度分数
            sim_scores = list(enumerate(cosine_similarities[idx]))
            
            # 按相似度降序排序
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # 获取前N个推荐（不包括自己）
            sim_scores = sim_scores[1:n+1]
            
            # 获取推荐的文章标题和相似度分数
            recommendations = []
            for i, score in sim_scores:
                title = data_df.index[i]
                recommendations.append((title, score))
            
            return recommendations
            
        except Exception as e:
            print(f"获取推荐时发生错误: {str(e)}")
            return []
    
    return get_recommendations

# 示例使用
if __name__ == "__main__":
    # 定义CSV文件路径
    csv_file_path = r'E:\知乎-AIGC-工程师\1.主干课\@4.14-Embeddings和向量数据库\3.作业\Health-Maintenance-Recommendations\health maintenance recommendations.csv'
    
    df = None  # 初始化df为None
    try:
        # 尝试使用 UTF-8 编码读取CSV文件
        print(f"尝试使用 UTF-8 编码读取文件: {csv_file_path}")
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        print("UTF-8 编码读取成功。")
    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_file_path}")
        exit()
    except UnicodeDecodeError:
        print("UTF-8 编码读取失败，尝试使用 GBK 编码...")
        try:
            # 如果 UTF-8 失败，尝试使用 GBK 编码
            df = pd.read_csv(csv_file_path, encoding='gbk')
            print("GBK 编码读取成功。")
        except Exception as e:
            print(f"使用 GBK 编码读取时也出错: {e}")
            print("请检查文件的编码格式，确保它是 UTF-8 或 GBK。")
            exit()
    except Exception as e:
        print(f"读取CSV文件时发生未知错误: {e}")
        exit()

    # 确保DataFrame已成功加载
    if df is None:
        print("未能成功加载数据，程序退出。")
        exit()

    # 数据探索（显示清理前的状态）
    print("\n=== 数据集探索（清理前）===")
    total_rows = len(df)
    print(f"数据集总行数：{total_rows}")
    
    # 检查空值
    null_counts = df.isnull().sum()
    print("\n各列的空值统计：")
    for column, null_count in null_counts.items():
        print(f"- {column}: {null_count} 个空值")
    
    print("\n数据集的所有列名：")
    for i, column in enumerate(df.columns, 1):
        print(f"{i}. {column}")

    # 数据清理
    print("\n=== 开始数据清理 ===")
    # 删除所有列都为空的行
    df_cleaned = df.dropna(how='all')
    # 删除必需列（文章标题和中心内容描述）中任一个为空的行
    df_cleaned = df_cleaned.dropna(subset=['文章标题', '中心内容描述'])
    
    # 显示清理后的数据状态
    print(f"\n=== 数据集探索（清理后）===")
    print(f"清理后的文章数量：{len(df_cleaned)}")
    print(f"清理后的唯一文章数：{df_cleaned['文章标题'].nunique()}")
    
    # 使用清理后的数据创建推荐系统
    print("\n=== 创建推荐系统 ===")
    recommend_health = create_health_recommendations(df_cleaned.copy())
    
    # 获取数据中的第一个文章标题用于测试推荐
    if not df_cleaned.empty:
        test_item_title = df_cleaned['文章标题'].iloc[5]
        print(f"\n基于'{test_item_title}'的推荐：")
        recommendations = recommend_health(test_item_title)
        if recommendations:
            for title, score in recommendations:
                print(f"- {title} (相似度: {score:.2f})")
        else:
            pass
    else:
        print("数据文件为空或缺少'文章标题'列，无法进行推荐测试。")

    # 分析高频词
    print("\n分析描述中的高频词：")
    valid_descriptions = df_cleaned['中心内容描述'].dropna()
    if not valid_descriptions.empty:
        common_words = get_top_n_words(valid_descriptions, 1, 10)
        if common_words:
            plot_top_words(common_words, '健康维护建议中的高频词Top10') 
        else:
            print("无法提取高频词。")
    else:
        print("描述列为空或无效，无法分析高频词。") 