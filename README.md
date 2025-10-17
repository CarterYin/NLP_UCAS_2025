# 这是中国科学院大学本科部2025NLP的作业
## Homework1（对应文件夹hw1）
### 题目
1.分别收集尽量多的英语和汉语文本，编写程序计算这些文本中英语字母和汉字的熵，对比本章课件第18页上表中给出的结果。然后逐步扩大文本规模，如每次增加固定的数量，如2M/5M等，重新计算文本规模扩大之的熵，分析多次增加之后熵的变化情况。

要求：

- 利用爬虫工具从互联网上收集样本，并对样本进行处理，如清洗乱码等；
- 设计算法并编程实现在收集样本上字母/汉字的概率和熵的计算；
- 当改变样本规模时重新计算字母/汉字的概率和熵, 并对比计算结果；
- 完成一份技术报告，在报告中写明利用什么爬虫工具从哪些网站上收集的样本，如何进行的样本清洗，清洗后样本的规模，在不同样本规模下计算的结果等。实验分析有较大的伸缩空间。

### 爬虫部分
**致谢：感谢https://github.com/caspiankexin/people-daily-crawler-date 仓库的爬虫，本仓库爬取人民日报部分内容皆为该仓库修改得来**

**主要实现了针对人民日报20241201之后的内容，以及Global Times主页内文章内容的爬取。**

对人民日报的爬虫的主要功能如下：

- 对2024年12月后的内容进行爬取（原仓库已经实现）
- 在命令行进行日期输入后自动化按照时间顺序爬取内容（原仓库已经实现）
- 实时监控爬取内容的大小和日期进度（新增加监控内容大小）
- 在2M，5M，10M的时候进行快照保存（新增加快照保存）
- 处理乱码和无意义的时间文本部分（新增加功能）

对Global Times的爬虫主要功能如下：

- 对Global Times主页的文章超链接进行访问，最大读取两层超链接（启发来自于Yu Shi）
- 忽略大量的标题类文本，中文文本，无关新闻的文本
- 在2M，5M，10M的时候进行快照保存

### 算法部分（概率和熵的计算和比较）

编写了程序compute_entropy.py，通过执行以下命令，可以看到统计结果。

```bash
python compute_entropy.py data\rmrb_snapshot_2MB.txt --top 10
```
```bash
python compute_entropy.py data\rmrb_snapshot_5MB.txt --top 10
```
```bash
python compute_entropy.py data\rmrb_snapshot_10MB.txt --top 10
```
统计结果示例如下：

```bash
(nlp) PS E:\homework\nlp\hw1> python compute_entropy.py data\rmrb_snapshot_2MB.txt --top 10
Corpus: data\rmrb_snapshot_2MB.txt
Total Chinese characters: 602680
Unique Chinese characters: 3571
Shannon entropy: 9.533166 bits

Top characters:
rank char     count    prob      information(bits)  contribution
   1  的      12996  0.021564        5.535253         0.119360
   2  国       7551  0.012529        6.318581         0.079166
   3  中       6743  0.011188        6.481858         0.072521
   4  一       5608  0.009305        6.747762         0.062789
   5  发       4736  0.007858        6.991579         0.054941
   6  新       4110  0.006820        7.196110         0.049074
   7  化       4006  0.006647        7.233086         0.048078
   8  人       3977  0.006599        7.243568         0.047799
   9  和       3966  0.006581        7.247564         0.047693
  10  业       3916  0.006498        7.265868         0.047211
```