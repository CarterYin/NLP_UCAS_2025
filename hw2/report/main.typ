#import "template.typ": *
#import "@preview/cram-snap:0.2.2": *
#set outline(title: "目录")

#show: bubble.with(
  title: "NLP第二次作业技术报告",
  author: "尹超",
  affiliation: "中国科学院大学\nUniversity of Chinese Academy of Sciences",
  // date: datetime.today().display(),
  subtitle: "基于FNN/RNN/LSTM的词向量学习与分析",
  date: datetime.today().display(),
  year: "2025",
  class: "2313AI",
  other: ("Template:https://github.com/hzkonor/bubble-template",
  "GitHub地址:https://github.com/CarterYin/NLP_UCAS_2025"),
  //main-color: "4DA6FF", //set the main color
  logo: image("../../cas_logo/CAS_logo.png"), //set the logo
  //"My GitHub:https://github.com/CarterYin"
) 

#outline()
#pagebreak()

// Edit this content to your liking

= Chapter 1 Introduction
本次作业内容如下：

1. (简答题)

  请下载调试FNN、RNN和LSTM模型的开源工具。利用北京大学标注的《人民日报》1998年1月份的分词语料，或者利用网络爬虫自己从互联网上收集足够多的英文文本语料，借助FNN或者RNN/LSTM开源工具，完成如下任务，并撰写一份实验报告：

  (1) 获得汉语或英语词语的词向量. 

  (2) 对于同一批词汇，对比分别用FNN,RNN或LSTM获得的词向量的差异。 

  (3) 利用你认为最好的词向量结果，对于随机选取的20个词汇分别计算与其词向量最相似的前10个单词，按相似度大小排序，人工对比排序结果是否与你的判断一致。

  (4) [选做]如果汉语和英语的词向量都学习到了，请对比同一个意思的汉语词汇和英语词汇，如“书”和‘book’，“工作”和‘work/ job’ 等，分析其向量距离。



说明

- 如果计算资源的限制，神经网络参数不必选择过大，例如：词表选择1000个左右单词即可, 其余单词用代替；词向量的维度可设为10左右；神经网络的层数设置为1到2层； 

- 可以使用某一种开放的深度学习框架，如TensorFlow或者PyTorch。

- 如果不借助开源工具和开放的深度学习框架，题目中的某些任务可以不做。



= Chapter 2 Data Collection and Cleaning
== 爬虫类型和语料来源网站
本次实验中，我采取了定向（垂直）爬虫（Focused Web Crawler），从人民日报和China Daily两个网站上分别爬取了中文和英文文本数据。

== 爬虫实现流程与细节
我编写了两个独立的爬虫脚本，分别命名为 rm.py（针对人民日报）和 cd.py（针对 China Daily）。

=== 爬虫实现流程
两份爬虫脚本都是面向具体电子报网站的定向采集工具，整体流程相近：通过日期驱动生成索引页 URL，逐级解析出文章列表，再抓取正文并累积到单一语料文件；在此基础上又引入 CorpusMonitor，按 2MB/5MB/10MB 阈值自动截取快照，便于规模控制和阶段分析。rm.py 依赖较固定的人民网上版面结构，层层定位到文章正文；cd.py 则针对 China Daily 的 HTML 差异做了更复杂的选择器和清洗逻辑（如多套候选节点、元信息剔除、重复段落去除），以适应版式变化。两者都使用 requests + BeautifulSoup 组合处理静态页面，并在请求层面设置通用的 UA/超时、在抓取循环中加延时，以降低被封风险，属于典型的模板化垂直爬虫实现。

=== 爬虫实现细节
细节处理方面，爬虫脚本均实现了断点续爬功能，利用本地文件记录已抓取日期，避免重复下载。文本清洗主要包括 HTML 标签剥离、空白字符归一化、非正文内容过滤、乱码去除等，确保语料纯净度。最终的语料文件均采用 UTF-8 编码保存，便于后续处理。

== 爬取结果
爬取结果方面，人民日报从20241201爬取到20250301，China Daily 则从20241201爬取到20250401。分别成功获得了三份不同大小的语料文件，大小为 2MB/5MB/10MB，均为纯文本格式，编码为 UTF-8。检查显示，文本内容较为干净，无乱码，符合后续处理要求。





= Chapter 4 Conclusion


= Appendix: Code Listings
代码仓库地址：https://github.com/CarterYin/NLP_UCAS_2025

为完整性起见，以下列出所有代码文件内容。
