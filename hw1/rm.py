'''
代码名称：爬取人民日报数据为txt文件
编写日期：2025年1月1日
作者：github（caspiankexin）（CarterYin）
版本：第3版
可爬取的时间范围：2024年12月起
注意：此代码仅供交流学习，不得作为其他用途。
'''


import requests
import bs4
import os
import datetime
import time
from typing import Optional, List

MB = 1024 * 1024
THRESHOLDS = [2 * MB, 5 * MB, 10 * MB]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')


class CorpusMonitor:
    """监控累计抓取体积并按阈值写入快照。"""

    def __init__(self, thresholds: Optional[List[int]] = None) -> None:
        self._thresholds = list(thresholds or THRESHOLDS)
        self._buffer = bytearray()
        self._total_bytes = 0

    def update(self, payload: bytes) -> None:
        self._buffer.extend(payload)
        self._total_bytes += len(payload)
        print(f"当前累计大小: {self._total_bytes / MB:.2f} MB")
        while self._thresholds and self._total_bytes >= self._thresholds[0]:
            threshold = self._thresholds.pop(0)
            self._save_snapshot(threshold)

    def _save_snapshot(self, threshold: int) -> None:
        os.makedirs(DATA_DIR, exist_ok=True)
        size_mb = threshold // MB
        filename = f"rmrb_snapshot_{size_mb}MB.txt"
        path = os.path.join(DATA_DIR, filename)
        with open(path, 'wb') as f:
            f.write(self._buffer[:threshold])
        print(f"快照已保存: {filename}")

def fetchUrl(url):
    '''
    功能：访问 url 的网页，获取网页内容并返回
    参数：目标网页的 url
    返回：目标网页的 html 内容
    '''

    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
    }

    r = requests.get(url,headers=headers)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    return r.text

def getPageList(year, month, day):
    '''
    功能：获取当天报纸的各版面的链接列表
    参数：年，月，日
    '''
    url = 'http://paper.people.com.cn/rmrb/pc/layout/' + year + month + '/' + day + '/node_01.html'
    html = fetchUrl(url)
    bsobj = bs4.BeautifulSoup(html,'html.parser')
    temp = bsobj.find('div', attrs = {'id': 'pageList'})
    if temp:
        pageList = temp.ul.find_all('div', attrs = {'class': 'right_title-name'})
    else:
        pageList = bsobj.find('div', attrs = {'class': 'swiper-container'}).find_all('div', attrs = {'class': 'swiper-slide'})
    linkList = []

    for page in pageList:
        link = page.a["href"]
        url = 'http://paper.people.com.cn/rmrb/pc/layout/'  + year + month + '/' + day + '/' + link
        linkList.append(url)

    return linkList

def getTitleList(year, month, day, pageUrl):
    '''
    功能：获取报纸某一版面的文章链接列表
    参数：年，月，日，该版面的链接
    '''
    html = fetchUrl(pageUrl)
    bsobj = bs4.BeautifulSoup(html,'html.parser')
    temp = bsobj.find('div', attrs = {'id': 'titleList'})
    if temp:
        titleList = temp.ul.find_all('li')
    else:
        titleList = bsobj.find('ul', attrs = {'class': 'news-list'}).find_all('li')
    linkList = []

    for title in titleList:
        tempList = title.find_all('a')
        for temp in tempList:
            link = temp["href"]
            if 'content' in link:
                url = 'http://paper.people.com.cn/rmrb/pc/content/' + year + month + '/' + day + '/' + link
                linkList.append(url)

    return linkList

def getContent(html):
    '''
    功能：解析 HTML 网页，获取新闻的文章内容
    参数：html 网页内容
    '''
    bsobj = bs4.BeautifulSoup(html,'html.parser')

    # 获取文章 标题
    title = bsobj.h3.text + '\n' + bsobj.h1.text + '\n' + bsobj.h2.text + '\n'
    #print(title)

    # 获取文章 内容
    pList = bsobj.find('div', attrs = {'id': 'ozoom'}).find_all('p')
    content = ''
    for p in pList:
        content += p.text + '\n'
    #print(content)

    # 返回结果 标题+内容
    resp = title + content
    return resp

def append_to_corpus(article_text, corpus_path, monitor: Optional[CorpusMonitor] = None):
    '''
    功能：将文章内容追加到单一语料文件
    参数：文章内容字符串，语料文件路径
    '''
    corpus_dir = os.path.dirname(corpus_path)
    if corpus_dir and not os.path.exists(corpus_dir):
        os.makedirs(corpus_dir)

    with open(corpus_path, 'a', encoding='utf-8') as f:
        f.write(article_text)

    if monitor is not None:
        monitor.update(article_text.encode('utf-8'))

def download_rmrb(year, month, day, corpus_path, monitor: Optional[CorpusMonitor] = None):
    '''
    功能：爬取《人民日报》网站 某年 某月 某日 的新闻内容，并追加保存到指定语料文件
    参数：年，月，日，语料文件路径
    '''
    pageList = getPageList(year, month, day)
    pageNo = 0
    for page in pageList:
        try:
            pageNo = pageNo + 1
            titleList = getTitleList(year, month, day, page)
            titleNo = 0
            for url in titleList:
                titleNo = titleNo + 1

                # 获取新闻文章内容
                html = fetchUrl(url)
                content = getContent(html)

#                article_header = (
#                    f"### 日期: {year}-{month}-{day} 版面: {str(pageNo).zfill(2)} 文章: {str(titleNo).zfill(2)}\n"
#                )
#                article_meta = f"来源链接: {url}\n"
                article_body = content.strip() + "\n\n"
                append_to_corpus(article_body, corpus_path, monitor)
        except Exception as e:
            print(f"日期 {year}-{month}-{day} 下的版面 {page} 出现错误：{e}")
            continue


def gen_dates(b_date, days):
    day = datetime.timedelta(days = 1)
    for i in range(days):
        yield b_date + day * i


def get_date_list(beginDate, endDate):
    """
    获取日期列表
    :param start: 开始日期
    :param end: 结束日期
    :return: 开始日期和结束日期之间的日期列表
    """

    start = datetime.datetime.strptime(beginDate, "%Y%m%d")
    end = datetime.datetime.strptime(endDate, "%Y%m%d")

    data = []
    for d in gen_dates(start, (end-start).days):
        data.append(d)

    return data


if __name__ == '__main__':
    '''
    主函数：程序入口
    '''
    # 输入起止日期，爬取之间的新闻
    print("欢迎使用人民日报爬虫，请输入以下信息：")
    beginDate = input('请输入开始日期:')
    endDate = input('请输入结束日期:')
    corpus_path = input("请输入语料文件的保存路径（例如 D:/data/rmrb_corpus.txt）：").strip()
    if not corpus_path:
        raise ValueError("语料文件路径不能为空")
    corpus_path = os.path.abspath(corpus_path)
    corpus_dir = os.path.dirname(corpus_path) or '.'
    if not os.path.exists(corpus_dir):
        os.makedirs(corpus_dir)
    with open(corpus_path, 'w', encoding='utf-8') as f:
        f.write('')
    monitor = CorpusMonitor()
    data = get_date_list(beginDate, endDate)

    for d in data:
        year = str(d.year)
        month = str(d.month) if d.month >=10 else '0' + str(d.month)
        day = str(d.day) if d.day >=10 else '0' + str(d.day)
        download_rmrb(year, month, day, corpus_path, monitor)
        print("爬取完成：" + year + month + day)
        time.sleep(5)        # 怕被封 IP 爬一爬缓一缓，爬的少的话可以注释掉

    lastend = input("本月数据爬取完成!可以关闭软件了")

