# 朴素贝叶斯邮件分类器

## 项目概述
本项目实现了一个基于朴素贝叶斯的邮件分类器，能够区分垃圾邮件和普通邮件。支持两种特征选择方法和样本平衡处理。

## 算法基础
采用多项式朴素贝叶斯分类器，基于贝叶斯定理：
$$ P(y|x) = \frac{P(x|y)P(y)}{P(x)} $$
其中特征之间条件独立假设成立。

## 数据处理流程
1. 从文本文件中加载邮件数据
2. 进行分词处理（自动由CountVectorizer/TfidfVectorizer完成）
3. 过滤停用词（自动完成）
4. 提取特征（高频词或TF-IDF加权）

## 特征构建方法
### 高频词特征
- 使用CountVectorizer统计词频
- 选择出现频率最高的1000个词作为特征
- 数学表达：$$ \text{count}(t,d) $$

### TF-IDF加权特征
- 使用TfidfVectorizer计算TF-IDF值
- 数学表达：$$ \text{tfidf}(t,d) = \text{tf}(t,d) \times \log(\frac{N}{\text{df}(t)}) $$
- 同样选择最重要的1000个特征

### 代码截图
<img src="https://github.com/ustin3/-3/blob/main/1.png" alt="图片描述" width = "800" height = "图片长度" />
<img src="https://github.com/ustin3/-3/blob/main/2.png" alt="图片描述" width = "800" height = "图片长度" />
<img src="https://github.com/ustin3/-3/blob/main/3.png" alt="图片描述" width = "800" height = "图片长度" />
<img src="https://github.com/ustin3/-3/blob/main/4.png" alt="图片描述" width = "800" height = "图片长度" />
