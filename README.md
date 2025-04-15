# 健康维护推荐系统

这是一个基于文本相似度的健康维护推荐系统，可以根据用户选择的健康活动推荐相似的其他健康维护方案。

## 功能特点

- 使用 `jieba` 进行中文分词处理，支持中英文
- 基于TF-IDF的文本特征提取，使用 1-gram 到 3-gram
- 使用余弦相似度计算推荐项目
- 支持高频词统计（默认 1-gram）和可视化
- 从 CSV 文件加载数据
- 数据清理：去除空值行

## 安装说明

1. 克隆项目到本地
2. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 使用方法

### 运行示例程序

直接运行 Python 文件即可查看示例输出：
```bash
python health_maintenance_recommendations.py
```
该程序会自动：
1. 从指定的 CSV 文件加载数据 (`health maintenance recommendations.csv`)。
2. 进行数据探索和清理。
3. 基于数据中的第一个项目进行推荐。
4. 分析描述中的高频词（默认 1-gram）并显示图表。

**如何修改测试推荐的文章标题？**

默认情况下，程序使用数据集中的第一个文章标题进行推荐测试。如果你想测试特定的文章标题，可以修改 `health_maintenance_recommendations.py` 文件中的大约第 332 行：

```python
# 将 iloc[0] 修改为你想要测试的文章标题字符串
# test_item_title = df_cleaned['文章标题'].iloc[0] # 这是默认值
test_item_title = "你想要测试的具体文章标题"
```
请确保你输入的标题存在于你的 CSV 文件中。

**如何调整 N-Gram 参数？**

代码中有两个地方使用了 N-Gram 参数，分别影响**推荐相似度计算**和**高频词分析**。

1.  **调整推荐相似度计算的 N-Gram (TF-IDF)**:
    *   **目的**: 影响文章之间相似度的计算精度。使用更高阶的 N-Gram（如 2-gram, 3-gram）可以捕捉更复杂的词语搭配和短语，可能提高推荐相关性，但也会增加计算量。
    *   **修改位置**: 在 `health_maintenance_recommendations.py` 文件中找到 `create_health_recommendations` 函数内部的 `TfidfVectorizer` 初始化部分（大约第 200 行）。
    *   **修改方法**: 修改 `ngram_range` 参数。例如，只使用 1-gram 和 2-gram：
        ```python
        tfidf = TfidfVectorizer(
            tokenizer=ChineseTokenizer(),
            ngram_range=(1, 2), # 原默认值为 (1, 3)
            min_df=0.01,
            max_df=0.95
        )
        ```

2.  **调整高频词分析的 N-Gram (CountVectorizer)**:
    *   **目的**: 分析不同长度词组的出现频率。默认只统计单个词（1-gram）。
    *   **修改位置**: 在 `health_maintenance_recommendations.py` 文件中找到 `get_top_n_words` 函数内部的 `CountVectorizer` 初始化部分（大约第 108 行）。
    *   **修改方法**: 修改 `ngram_range` 参数。例如，只分析 2-gram（双词组合）的频率：
        ```python
        vec = CountVectorizer(
            tokenizer=tokenizer,
            ngram_range=(2, 2), # 原默认值为 (n, n)，n默认为1
            min_df=2,
            max_df=0.95
        )
        ```
    *   **注意**: 如果你在主程序块 (`if __name__ == "__main__":`) 中调用 `get_top_n_words`，你可能还需要修改函数调用时的 `n` 参数（大约第 343 行），并相应调整图表标题：
        ```python
        # 获取 Top 10 的 2-gram
        common_words = get_top_n_words(valid_descriptions, n=2, k=10)
        if common_words:
            plot_top_words(common_words, '健康维护建议中的高频词Top10 (2-gram)') 
        ```

## 文件说明

- `health_maintenance_recommendations.py`: 主程序文件，包含核心逻辑和函数
- `requirements.txt`: 依赖包列表
- `README.md`: 项目说明文档

## 系统要求

- Python 3.7+
- 依赖库 (pandas, scikit-learn, matplotlib, numpy, jieba)
- 支持中文显示的操作系统（用于可视化） 
