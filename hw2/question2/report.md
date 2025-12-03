# 词向量模型对比报告 (FNN vs RNN vs LSTM)

## 中文 (Chinese) 语料分析

### 1. 词义相似度定性分析 (Nearest Neighbors)

选取了几个高频关键词，分别列出它们在不同模型下的 Top-5 近义词，以观察模型捕捉语义的能力差异。

#### 关键词: **中国**

| Rank | FNN | RNN | LSTM |
|---|---|---|---|
| 1 | 大使馆 (0.321) | 俄 (0.346) | 沿 (0.345) |
| 2 | 天天 (0.308) | 途径 (0.329) | 东方 (0.311) |
| 3 | 中亚 (0.303) | 研究所 (0.324) | 傅 (0.310) |
| 4 | 张家口 (0.294) | 超越 (0.323) | 大庆 (0.310) |
| 5 | 广西 (0.294) | 离开 (0.319) | 判断 (0.292) |


#### 关键词: **发展**

| Rank | FNN | RNN | LSTM |
|---|---|---|---|
| 1 | 树立 (0.370) | ２月 (0.365) | 爱心 (0.358) |
| 2 | 增加 (0.327) | 海峡 (0.348) | 波罗的海 (0.345) |
| 3 | 自己 (0.319) | 封闭 (0.336) | 中介 (0.330) |
| 4 | 藏 (0.295) | 除 (0.319) | 限制 (0.297) |
| 5 | 实力 (0.295) | 资料 (0.306) | 印尼 (0.292) |


#### 关键词: **经济**

| Rank | FNN | RNN | LSTM |
|---|---|---|---|
| 1 | 外交 (0.433) | 眼下 (0.366) | 高速 (0.393) |
| 2 | 产业 (0.414) | 副食品 (0.325) | 宣传 (0.362) |
| 3 | 深化 (0.305) | 农业 (0.316) | 深化 (0.346) |
| 4 | 通讯 (0.299) | 生物 (0.304) | 资源 (0.308) |
| 5 | 学子 (0.284) | 营造 (0.284) | 不断 (0.308) |


#### 关键词: **人民**

| Rank | FNN | RNN | LSTM |
|---|---|---|---|
| 1 | 总之 (0.343) | 局势 (0.357) | 完善 (0.335) |
| 2 | 证据 (0.330) | 美好 (0.346) | 次 (0.306) |
| 3 | 布局 (0.328) | 党支部 (0.320) | ４日 (0.305) |
| 4 | 权限 (0.315) | 人间 (0.318) | 入 (0.300) |
| 5 | 从未 (0.314) | 坚持不懈 (0.314) | 大家 (0.297) |


#### 关键词: **希望**

| Rank | FNN | RNN | LSTM |
|---|---|---|---|
| 1 | 富有 (0.356) | 点 (0.358) | 胜利 (0.398) |
| 2 | 推动 (0.339) | 应急 (0.330) | 增强 (0.354) |
| 3 | 回归 (0.324) | 建华 (0.300) | 句 (0.333) |
| 4 | 统一战线 (0.319) | 科技 (0.297) | 东京 (0.317) |
| 5 | 有关 (0.319) | 徘徊 (0.292) | 气氛 (0.316) |


### 2. 模型一致性定量分析 (Model Consistency)

计算不同模型之间 Top-10 近义词的重叠度 (Jaccard Similarity)，以评估模型学习到的语义空间的一致性。

| 模型对比 | 平均 Jaccard 相似度 (Top-10 Neighbors) |
|---|---|
| FNN vs RNN | 0.0043 |
| RNN vs LSTM | 0.0037 |
| FNN vs LSTM | 0.0032 |


## 英文 (English) 语料分析

### 1. 词义相似度定性分析 (Nearest Neighbors)

选取了几个高频关键词，分别列出它们在不同模型下的 Top-5 近义词，以观察模型捕捉语义的能力差异。

#### 关键词: **china**

| Rank | FNN | RNN | LSTM |
|---|---|---|---|
| 1 | germany (0.373) | instance (0.364) | africa (0.376) |
| 2 | suzhou (0.346) | hunan (0.352) | japan (0.354) |
| 3 | indonesia (0.341) | let (0.344) | fusion (0.323) |
| 4 | stakeholders (0.333) | canada (0.333) | country (0.316) |
| 5 | nation (0.315) | unchanged (0.317) | vietnam (0.316) |


#### 关键词: **development**

| Rank | FNN | RNN | LSTM |
|---|---|---|---|
| 1 | 82 (0.333) | innovation (0.387) | solely (0.338) |
| 2 | sea (0.330) | myself (0.340) | maintenance (0.305) |
| 3 | sentiment (0.314) | give (0.319) | principles (0.303) |
| 4 | interest (0.271) | interests (0.294) | safety (0.301) |
| 5 | command (0.267) | user (0.282) | rest (0.301) |


#### 关键词: **world**

| Rank | FNN | RNN | LSTM |
|---|---|---|---|
| 1 | luckin (0.339) | anti (0.325) | healthy (0.348) |
| 2 | aging (0.338) | showing (0.278) | comment (0.332) |
| 3 | minerals (0.331) | impressed (0.274) | scale (0.317) |
| 4 | planned (0.328) | thursday (0.273) | congress (0.314) |
| 5 | strait (0.318) | promotes (0.270) | infrastructure (0.314) |


#### 关键词: **cooperation**

| Rank | FNN | RNN | LSTM |
|---|---|---|---|
| 1 | mexico (0.421) | relationship (0.367) | industrialization (0.364) |
| 2 | savings (0.355) | ton (0.364) | services (0.321) |
| 3 | partnerships (0.346) | credibility (0.335) | fu (0.311) |
| 4 | trips (0.345) | friendship (0.328) | 38 (0.307) |
| 5 | trading (0.336) | multilateralism (0.308) | interview (0.301) |


#### 关键词: **peace**

| Rank | FNN | RNN | LSTM |
|---|---|---|---|
| 1 | citizens (0.325) | figure (0.385) | designs (0.341) |
| 2 | union (0.321) | solely (0.364) | railways (0.306) |
| 3 | influential (0.313) | opinions (0.344) | volume (0.304) |
| 4 | ambitious (0.303) | barrier (0.338) | decarbonization (0.303) |
| 5 | intellectual (0.300) | ideas (0.331) | america (0.293) |


### 2. 模型一致性定量分析 (Model Consistency)

计算不同模型之间 Top-10 近义词的重叠度 (Jaccard Similarity)，以评估模型学习到的语义空间的一致性。

| 模型对比 | 平均 Jaccard 相似度 (Top-10 Neighbors) |
|---|---|
| FNN vs RNN | 0.0026 |
| RNN vs LSTM | 0.0037 |
| FNN vs LSTM | 0.0026 |

