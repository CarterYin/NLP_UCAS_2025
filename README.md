# 这是中国科学院大学本科部2025NLP的作业
## Homework1（对应文件夹hw1）
### 题目
1.分别收集尽量多的英语和汉语文本，编写程序计算这些文本中英语字母和汉字的熵，对比本章课件第18页上表中给出的结果。然后逐步扩大文本规模，如每次增加固定的数量，如2M/5M等，重新计算文本规模扩大之的熵，分析多次增加之后熵的变化情况。

要求：

- 利用爬虫工具从互联网上收集样本，并对样本进行处理，如清洗乱码等；
- 设计算法并编程实现在收集样本上字母/汉字的概率和熵的计算；
- 当改变样本规模时重新计算字母/汉字的概率和熵, 并对比计算结果；
- 完成一份技术报告，在报告中写明利用什么爬虫工具从哪些网站上收集的样本，如何进行的样本清洗，清洗后样本的规模，在不同样本规模下计算的结果等。实验分析有较大的伸缩空间。

### 环境
创建虚拟环境
```bash
conda create -n nlp python=3.13 -y
```
下载必要的包
```bash
pip install requests beautifulsoup4
```


### 爬虫部分
**致谢：感谢https://github.com/caspiankexin/people-daily-crawler-date 仓库的爬虫，本仓库爬取人民日报部分内容皆为该仓库修改得来**

**主要实现了针对人民日报20241201之后的内容，以及Global Times主页内文章内容的爬取。**

#### 对人民日报的爬虫的主要功能如下(rm.py)：

- 对2024年12月后的内容进行爬取（原仓库已经实现），在本次计算比较中，我采用了爬取20241201-20250301的内容，实际上发现爬取到20250106已经满足10M。
- 在命令行进行日期输入后自动化按照时间顺序爬取内容（原仓库已经实现）。
- 实时监控爬取内容的大小和日期进度（新增加监控内容大小）。
- 在2M，5M，10M的时候进行快照保存（新增加快照保存）。
- 处理乱码和无意义的时间文本部分（新增加功能）。

示例交互命令：
```bash
(nlp) PS E:\homework\nlp\hw1> python rm.py
欢迎使用人民日报爬虫，请输入以下信息：
请输入开始日期:20241201
请输入结束日期:20250301
请输入语料文件的保存路径（例如 D:/data/rmrb_corpus.txt）：data/rm.txt
```



#### 对China Daily的爬虫主要功能如下(cd.py)：

- 和rm.py一样的功能，跳过了空白文章和无法索引的文章。
- 在2M，5M，10M的时候进行快照保存。

示例交互命令：
```bash
(nlp) PS E:\homework\nlp\hw1> python cd.py    
欢迎使用 China Daily 英文版爬虫，请输入以下信息：
请输入开始日期:20241201
请输入结束日期:20250401
请输入语料文件的保存路径（例如 D:/data/chinadaily_corpus.txt）：data/cd.txt
```


### 算法部分（概率和熵的计算和比较）

#### 中文部分（人民日报）
编写了程序compute_rmrb.py，通过执行以下命令，可以看到人民日报中，中文字符的统计结果。

**compute_rmrb.py 里统计字符的入口是 iter_cjk_chars，它仅在字符的 Unicode 码点落在 0x4E00 到 0x9FFF（CJK Unified Ideographs 区间）时才 yield，所以数字、英文字母和其他非汉字符号都会被直接过滤掉，不会进入计数或熵计算。**

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

```bash
(nlp) PS E:\homework\nlp\hw1> python compute_entropy.py data\rmrb_snapshot_5MB.txt --top 10
Corpus: data\rmrb_snapshot_5MB.txt
Total Chinese characters: 1503549
Unique Chinese characters: 4307
Shannon entropy: 9.639168 bits

Top characters:
rank char     count    prob      information(bits)  contribution
   1  的      33439  0.022240        5.490696         0.122113
   2  国      16802  0.011175        6.483595         0.072453
   3  中      15766  0.010486        6.575411         0.068949
   4  一      13521  0.008993        6.797026         0.061124
   5  发      11182  0.007437        7.071050         0.052588
   6  人      10897  0.007248        7.108297         0.051518
   7  业       9574  0.006368        7.295034         0.046452
   8  和       9550  0.006352        7.298655         0.046358
   9  在       9487  0.006310        7.308204         0.046113
  10  年       9333  0.006207        7.331815         0.045511
```

```bash
(nlp) PS E:\homework\nlp\hw1> python compute_entropy.py data\rmrb_snapshot_10MB.txt --top 10
Corpus: data\rmrb_snapshot_10MB.txt
Total Chinese characters: 3006557
Unique Chinese characters: 4735
Shannon entropy: 9.631801 bits

Top characters:
rank char     count    prob      information(bits)  contribution
   1  的      66909  0.022254        5.489768         0.122171
   2  国      32913  0.010947        6.513311         0.071302
   3  中      30465  0.010133        6.624816         0.067128
   4  一      26290  0.008744        6.837454         0.059788
   5  人      22374  0.007442        7.070145         0.052614
   6  发      21741  0.007231        7.111550         0.051425
   7  业      20417  0.006791        7.202198         0.048909
   8  和      19202  0.006387        7.290712         0.046564
   9  在      18814  0.006258        7.320162         0.045807
  10  年      18745  0.006235        7.325463         0.045672
```




#### 英文部分（China Daily）
编写了程序compute_cd.py，通过执行以下命令，可以看到China Daily中，英文字符的统计结果。（爬取的是从20241201-20250401的文章）
```bash
python compute_cd.py data\cd_snapshot_2MB.txt --top 10
```
```bash
python compute_cd.py data\cd_snapshot_5MB.txt --top 10
```
```bash
python compute_cd.py data\cd_snapshot_10MB.txt --top 10
```



统计结果示例如下：

```bash
(nlp) PS E:\homework\nlp\hw1> python compute_cd.py data\cd_snapshot_2MB.txt --top 10
Corpus: data\cd_snapshot_2MB.txt
Total letters: 1706620
Unique letters: 26
Shannon entropy: 4.162821 bits

Top letters:
rank char    count    prob       information(bits) contribution
   1  e     200370  0.117408        3.090403         0.362837
   2  t     149595  0.087656        3.512008         0.307848
   3  i     144565  0.084708        3.561352         0.301676
   4  a     143569  0.084125        3.571326         0.300437
   5  n     141317  0.082805        3.594135         0.297613
   6  o     125017  0.073254        3.770946         0.276237
   7  s     111119  0.065111        3.940964         0.256598
   8  r     107151  0.062786        3.993425         0.250729
   9  h      73296  0.042948        4.541264         0.195038
  10  l      68587  0.040189        4.637063         0.186358
```

```bash
(nlp) PS E:\homework\nlp\hw1> python compute_cd.py data\cd_snapshot_5MB.txt --top 10
Corpus: data\cd_snapshot_5MB.txt
Total letters: 4264958
Unique letters: 26
Shannon entropy: 4.162419 bits

Top letters:
rank char    count    prob      information(bits)  contribution
   1  e     500480  0.117347        3.091147         0.362737
   2  t     373307  0.087529        3.514097         0.307585
   3  i     361563  0.084775        3.560213         0.301818
   4  a     360869  0.084613        3.562984         0.301473
   5  n     352718  0.082701        3.595944         0.297390
   6  o     311461  0.073028        3.775408         0.275710
   7  s     279295  0.065486        3.932670         0.257535
   8  r     268068  0.062854        3.991861         0.250903
   9  h     182403  0.042768        4.547330         0.194479
  10  l     170040  0.039869        4.648585         0.185335
```

```bash
(nlp) PS E:\homework\nlp\hw1> python compute_cd.py data\cd_snapshot_10MB.txt --top 10
Corpus: data\cd_snapshot_10MB.txt
Total letters: 8532582
Unique letters: 26
Shannon entropy: 4.162832 bits

Top letters:
rank char count prob information(bits) contribution
   1  e    1000965  0.117311        3.091591         0.362677
   2  t     746638  0.087504        3.514502         0.307534
   3  i     723501  0.084793        3.559915         0.301855
   4  a     721101  0.084511        3.564709         0.301259
   5  n     704407  0.082555        3.598501         0.297074
   6  o     619315  0.072582        3.784237         0.274669
   7  s     562327  0.065903        3.923501         0.258572
   8  r     537203  0.062959        3.989443         0.251171
   9  h     362041  0.042430        4.558757         0.193430
  10  l     341302  0.040000        4.643862         0.185754
```


