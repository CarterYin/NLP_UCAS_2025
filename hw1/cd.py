'''
代码名称：爬取 China Daily 英文版数据为 txt 文件
编写日期：2025年1月1日
作者：github（CarterYin）
版本：第1版
可爬取的时间范围：2021年起（以站点可用为准）
注意：此代码仅供交流学习，不得作为其他用途。
'''

import datetime
import os
import re
import time
import urllib.parse
from collections import OrderedDict
from typing import List, Optional

import bs4
import requests

MB = 1024 * 1024
THRESHOLDS = [2 * MB, 5 * MB, 10 * MB]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
BASE_DOMAIN = 'https://epaper.chinadaily.com.cn'
DEFAULT_EDITION = os.environ.get('CHINADAILY_EDITION', 'global').strip() or 'global'


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
        filename = f"cd_snapshot_{size_mb}MB.txt"
        path = os.path.join(DATA_DIR, filename)
        with open(path, 'wb') as f:
            f.write(self._buffer[:threshold])
        print(f"快照已保存: {filename}")


def fetch_url(url: str) -> str:
    """访问指定 URL 并返回 HTML 文本。"""
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or resp.encoding
    return resp.text


def build_index_url(year: str, month: str, day: str, edition: str) -> str:
    """构造某一天 China Daily 电子报索引页 URL。"""
    return f"{BASE_DOMAIN}/{edition}/{year}-{month}/{day}/index.html"


def extract_issue_links(index_html: str, year: str, month: str, day: str) -> List[str]:
    """从索引页提取当日所有版块的文章链接。"""
    soup = bs4.BeautifulSoup(index_html, 'html.parser')
    issue_path = f"/a/{year}{month}/{day}/"
    links = OrderedDict()

    for anchor in soup.find_all('a', href=True):
        href = anchor['href'].strip()
        if not href:
            continue
        if href.startswith('//'):
            href = 'https:' + href
        elif href.startswith('/'):
            href = urllib.parse.urljoin(BASE_DOMAIN, href)
        elif not href.startswith('http'):
            href = urllib.parse.urljoin(BASE_DOMAIN, href)
        if issue_path not in href:
            continue
        if not href.endswith('.html'):
            continue
        links[href] = None

    return list(links.keys())


def _first_non_empty_text(soup: bs4.BeautifulSoup, selectors: List[str]) -> Optional[str]:
    """按顺序返回第一个匹配选择器并含非空文本的节点内容。"""
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            text = node.get_text(' ', strip=True)
            if text:
                return text
    return None


def extract_article_content(html: str) -> str:
    """解析正文页面并返回标题及正文拼接文本。"""
    soup = bs4.BeautifulSoup(html, 'html.parser')

    article_root = None
    for selector in ['article', '#Content', '.content', '.text', '.lft', '.main_l', '.left', '.news', '.news_l']:
        article_root = soup.select_one(selector)
        if article_root:
            break
    if article_root is None:
        article_root = soup.body or soup

    headline = _first_non_empty_text(article_root, [
        'h1.headline',
        'h1.title',
        'div.lft h1',
        '#Content h1',
        'article h1',
        'h1',
    ])
    sub_head = _first_non_empty_text(article_root, [
        'h2.subhead',
        'h2.subtitle',
        'div.lft h2',
        '#Content h2',
        'article h2',
        'h2',
    ])

    meta_text = None
    meta_candidates = article_root.select('div.info, div.meta, p.meta, p.info, div.editor, div.source, p.source')
    for candidate in meta_candidates:
        candidate_text = candidate.get_text(' ', strip=True)
        if not candidate_text:
            continue
        if 'Updated' in candidate_text or candidate_text.startswith('By '):
            meta_text = candidate_text
            candidate.extract()
            break

    paragraphs: List[str] = []
    for block in article_root.find_all(['p', 'div']):
        if block.name == 'div' and block.find(['p', 'div']):
            continue
        text = block.get_text(' ', strip=True)
        if not text:
            continue
        if re.search(r'Copyright\s+1995', text):
            continue
        if 'Additional Links' in text or 'Search' == text:
            continue
        if text.upper().startswith('HOME') and 'Newspaper' in text:
            continue
        paragraphs.append(text)

    if not paragraphs:
        fallback_paragraphs = []
        for tag in soup.find_all('p'):
            text = tag.get_text(' ', strip=True)
            if len(text.split()) < 3:
                continue
            if re.search(r'Copyright\s+1995', text):
                continue
            fallback_paragraphs.append(text)
        paragraphs = fallback_paragraphs

    unique_paragraphs = list(OrderedDict((para, None) for para in paragraphs).keys())

    header_parts = [part for part in [headline, sub_head, meta_text] if part]
    body_text = '\n'.join(unique_paragraphs)
    combined = '\n'.join(header_parts + ([body_text] if body_text else []))
    return combined.strip()


def append_to_corpus(article_text: str, corpus_path: str, monitor: Optional[CorpusMonitor] = None) -> None:
    """将文章内容追加到单一语料文件。"""
    corpus_dir = os.path.dirname(corpus_path)
    if corpus_dir and not os.path.exists(corpus_dir):
        os.makedirs(corpus_dir)

    with open(corpus_path, 'a', encoding='utf-8') as f:
        f.write(article_text)

    if monitor is not None:
        monitor.update(article_text.encode('utf-8'))


def download_chinadaily(year: str, month: str, day: str, corpus_path: str, monitor: Optional[CorpusMonitor] = None, edition: str = DEFAULT_EDITION) -> None:
    """爬取 China Daily 网站指定日期的新闻并写入语料。"""
    index_url = build_index_url(year, month, day, edition)
    try:
        index_html = fetch_url(index_url)
    except Exception as exc:  # pragma: no cover - 网络异常提醒
        print(f"无法获取 {year}-{month}-{day} 的索引页 {index_url}: {exc}")
        return

    article_links = extract_issue_links(index_html, year, month, day)
    if not article_links:
        print(f"未在 {index_url} 找到文章链接。")
        return

    for seq, url in enumerate(article_links, 1):
        try:
            article_html = fetch_url(url)
            content = extract_article_content(article_html)
            if not content:
                print(f"跳过空白文章: {url}")
                continue
            article_body = content.strip() + "\n\n"
            append_to_corpus(article_body, corpus_path, monitor)
            print(f"成功抓取第 {seq:02d} 篇: {url}")
            time.sleep(1)
        except Exception as exc:  # pragma: no cover - 逐篇抓取问题提示
            print(f"文章 {url} 抓取失败: {exc}")
            continue


def gen_dates(b_date: datetime.datetime, days: int):
    day = datetime.timedelta(days=1)
    for i in range(days):
        yield b_date + day * i


def get_date_list(begin_date: str, end_date: str):
    """获取日期列表。"""
    start = datetime.datetime.strptime(begin_date, "%Y%m%d")
    end = datetime.datetime.strptime(end_date, "%Y%m%d")

    data = []
    for d in gen_dates(start, (end - start).days):
        data.append(d)

    return data


if __name__ == '__main__':
    print("欢迎使用 China Daily 英文版爬虫，请输入以下信息：")
    begin_date = input('请输入开始日期:')
    end_date = input('请输入结束日期:')
    corpus_path = input('请输入语料文件的保存路径（例如 D:/data/chinadaily_corpus.txt）：').strip()
    if not corpus_path:
        raise ValueError('语料文件路径不能为空')
    corpus_path = os.path.abspath(corpus_path)
    corpus_dir = os.path.dirname(corpus_path) or '.'
    if not os.path.exists(corpus_dir):
        os.makedirs(corpus_dir)
    with open(corpus_path, 'w', encoding='utf-8') as f:
        f.write('')
    monitor = CorpusMonitor()
    date_list = get_date_list(begin_date, end_date)

    for current in date_list:
        year = str(current.year)
        month = str(current.month) if current.month >= 10 else '0' + str(current.month)
        day = str(current.day) if current.day >= 10 else '0' + str(current.day)
        download_chinadaily(year, month, day, corpus_path, monitor, DEFAULT_EDITION)
        print("爬取完成：" + year + month + day)
        time.sleep(5)

    input('本段数据抓取完成!可以关闭软件了')
