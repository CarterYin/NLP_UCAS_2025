# 词向量相似度人工对比分析报告

本报告基于 LSTM 模型训练得到的词向量（中文与英文），随机选取了 20 个词汇，计算其最相似的 Top-10 单词，并进行了人工对比分析。

## 1. 中文词向量分析 (Chinese Embeddings)

| 词汇 (Word) | Top-10 相似词 (Neighbors) | 人工分析与判断 (Analysis) |
|---|---|---|
| **财政** | 文字, 使馆, 科委, 科研, １４, 无论是, 彭, 战略性, 汇, 签订 | **一般**。关联性较弱，"汇"和"签订"稍微沾边，但"文字"、"使馆"等关联度不高。可能是语料中"财政"常出现在政府报告中，与"科委"、"使馆"等机构名词并列出现。 |
| **以** | 创建, 官兵, 讯, 变, 回到, 公益, 分为, 征收, 消防, 谭 | **较差**。"以"是介词，语义模糊，上下文依赖强，因此相似词多为动词或名词，缺乏明显的语义聚类。 |
| **条件** | 风貌, 需要, 树, 公安, 十四大, １９９８, 公款, 石, 服装, 追 | **较差**。未见明显同义词或相关词（如"环境"、"基础"等），"需要"可能在句法上常搭配（"需要...条件"）。 |
| **教师** | 各国, 货币化, 灾民, 消防, 联欢, 浮动, 类, 奋力, 阿根廷, 光荣 | **较差**。未出现"学生"、"学校"、"教育"等强相关词。可能是数据量不足导致共现模式未被充分学习。 |
| **规模** | 分配, 副食品, 新年, 优势, 大大, 永远, 土耳其, 历史性, 城市, 片 | **一般**。"分配"、"城市"在宏观经济语境下可能相关，但整体噪声较大。 |
| **意识** | 状况, 干部, 喜讯, 排, 社会, 公正, 扩张, 没, 安理会, 一方面 | **一般**。"干部"、"社会"常与"意识"（如"公仆意识"、"社会意识"）搭配，反映了语料的政治新闻特性。 |
| **青年** | 入手, 评选, 空气, 男, 关公, 想起, 航空, 材料, 有利于, 石家庄 | **较差**。除了"男"可能指代性别外，其余词汇关联度极低。 |
| **仍** | 年轻人, 一度, 政党, 也, 平, 实际上, 又, 外国, 企业, 基因 | **一般**。"仍"是副词，"又"、"也"、"实际上"也是副词或连接词，词性上具有一定的一致性。 |
| **７日** | 重要性, ５日, ２８, 揭示, 事业, 合并, 批准, 严寒, 另一方面, 交易日 | **较好**。出现了"５日"、"２８"（可能指日期）、"交易日"，捕捉到了时间或日期的语义特征。 |
| **才** | 不仅, 无产阶级, 只, 逐渐, 国情, 色彩, 人心, 潜力, 曾经, 常务 | **一般**。"只"、"不仅"是关联词/副词，与"才"在句法功能上相似。 |
| **统一** | 县委, 顽强, 历史性, 停止, ’, 钻, 重建, 扶贫, 跳, 坚强 | **一般**。"历史性"（如"历史性统一"）、"重建"可能在特定语境下共现。 |
| **晚** | 至此, 奉献, 沈阳, 主人, 京, 范围, 为了, 于是, 大都, 大学生 | **较差**。未见"早"、"夜"、"时间"等相关词。 |
| **生产** | 猪, 健全, 外商, 取, 送给, 具, 网, 医疗队, 资源, 比例 | **一般**。"猪"（农业生产）、"资源"、"外商"（生产合作）有一定关联，反映了新闻语料中"生产"的多样化语境。 |
| **国防** | 献给, 红薯, 震撼, 助, 出生, 吕, 保证, 账户, 减少, 保障 | **一般**。"保障"（国防保障）有一定语义关联，其余较弱。 |
| **除** | 表演, 约, 遏制, 补充, 满, 具有, 上市, 女性, ２６, 久 | **较差**。作为介词或动词，语义难以捕捉。 |
| **●** | 涉及, 已, 也, ，, 早日, 爱好者, 约旦河, 金, 工商局, 驻军 | **无意义**。符号，通常作为列表项或分隔符，周围词汇随机性大。 |
| **会议** | 揭晓, 增幅, 住房, 艺术家, 市长, 处长, 团拜会, 女士, 协定, 思路 | **较好**。"团拜会"是会议的一种，"市长"、"处长"是参会人员，"协定"是会议成果，关联性尚可。 |
| **文明** | 舆论, 管理, 探测器, 网, 一生, 又, 客车, ２０时, 增值, 老人 | **较差**。"舆论"可能相关（精神文明），但整体较乱。 |
| **本** | 离退休, 有关, 罗, 党政, 花园, 造船, 吕梁, 检验, 安置, 充满 | **较差**。指代词或量词，语义依赖上下文。 |
| **怎么** | 读, 下半年, 敢, 临近, 比较, 村民, 调节, 表面, 连连, 作家 | **较差**。疑问代词，难以通过简单上下文捕捉同义词。 |

**中文总结**：
整体效果一般。模型能够捕捉到部分词性特征（如副词匹配副词）和特定语境下的共现关系（如"会议"与"团拜会"），但对于大多数实词，未能找到高质量的同义词。这主要是因为训练语料（1998年人民日报）相对较小，且 LSTM 模型训练时间（Epochs）可能不足，导致语义空间尚未充分收敛。

## 2. 英文词向量分析 (English Embeddings)

| Word | Top-10 Neighbors | Analysis |
|---|---|---|
| **asian** | suzhou, central, municipal, african, marketing, toy, healthcare, span, practices, fewer | **Good**. "african" is a similar demonym/adjective. "suzhou" is an Asian city. Captures some geographic/demographic context. |
| **support** | facilitate, emphasizing, responsible, teach, standard, sports, deliver, demonstrated, thailand, plans | **Fair**. "facilitate" is semantically related to support (help/enable). |
| **previous** | five, significantly, chip, percent, human, brussels, feature, immediate, patent, proud | **Poor**. Mostly unrelated. |
| **research** | robotics, india, academy, nezha, information, atlantic, autonomous, fundamental, author, concert | **Good**. "robotics", "academy", "information", "autonomous" are all strongly related to research topics. |
| **28** | cup, every, headquarters, 21, cheng, artifacts, unilateral, press, 23, 34 | **Good**. "21", "23", "34" are numbers. Captures the category of numerals. |
| **information** | tea, nurture, fraud, joint, 32, remarks, chip, research, oil, 85 | **Fair**. "research" and "chip" (tech) are related. "tea" is noise. |
| **years** | mean, porcelain, although, decade, xinjiang, contributes, height, guangdong, benefited, committee | **Good**. "decade" is a direct synonym/related time unit. |
| **though** | settle, alongside, heard, accused, railway, decarbonization, told, ball, replace, continent | **Poor**. Conjunction, hard to capture. |
| **tea** | fish, information, 2000, modern, computer, numerous, 19, chinese, wine, consultations | **Fair**. "wine" and "fish" are food/drink items. "chinese" is a strong collocate (Chinese tea). |
| **double** | accepted, improve, exceeding, deliver, january, coal, attend, c, evolved, maximum | **Poor**. |
| **ecosystem** | baby, seem, partner, courses, markets, soup, tons, strides, laboratory, meals | **Poor**. "laboratory" maybe related to scientific ecosystem, but "soup"/"baby" are noise. |
| **anniversary** | part, freight, undergone, half, regardless, factories, middle, carry, 61, sidelines | **Poor**. |
| **surge** | loans, specializing, participated, shows, placed, deaths, reflected, engaging, turbulence, realities | **Fair**. "loans" and "deaths" are things that can surge. |
| **time** | seeds, fill, war, avoiding, consecutive, 66, begun, achievements, champion, second | **Fair**. "second" is a unit of time. |
| **night** | republican, leveraged, while, milestone, positive, calligraphy, shipments, hospitals, tan, roughly | **Poor**. |
| **infrastructure** | class, reaffirmed, crises, world, flights, opinions, organizations, nepal, ethiopia, whether | **Fair**. "flights" (transport) is related to infrastructure. |
| **brands** | employees, movie, everywhere, relationships, technique, notable, 2028, layout, stations, collaboration | **Poor**. |
| **before** | met, pandemic, another, doctor, prioritizes, has, shop, portuguese, shan, institution | **Fair**. "met" (past tense) often happens *before*. "pandemic" is a time marker (before pandemic). |
| **our** | eco, false, rich, exotic, dark, blood, constructive, s, renewed, the | **Poor**. Pronoun. |
| **various** | artistic, employed, xiamen, enterprise, leader, bottlenecks, cantonese, million, related, code | **Poor**. |

**英文总结**：
英文结果略好于中文，特别是在名词和数字的聚类上。例如 "research" 关联了 "robotics", "academy"；"years" 关联了 "decade"；"28" 关联了其他数字。这表明模型捕捉到了部分语义类别。然而，对于抽象词汇和功能词，效果依然不理想。

## 3. 总体结论

通过人工对比，排序结果与人类直觉的**一致性较低**。

1.  **语义聚类初步显现**：在部分具体名词（如"research"、"7日"）上，模型能够找到同类词或相关词。
2.  **噪声较大**：大多数词汇的近义词列表中包含了大量无关词汇，表明词向量的质量有待提高。
3.  **原因分析**：
    *   **数据量**：10MB 左右的语料对于训练高质量词向量（通常需要 GB 级别）来说偏少。
    *   **模型参数**：为了演示目的，训练轮数（Epochs）较少，模型可能欠拟合。
    *   **上下文窗口**：N-gram 或 序列长度较短，限制了模型捕捉长距离依赖的能力。

**改进建议**：增加训练数据量，增加训练轮数，或使用更复杂的预训练模型（如 Word2Vec, GloVe, BERT）可以显著提升效果。
