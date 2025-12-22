import requests
from bs4 import BeautifulSoup
import os
import time
import random
import re
from datetime import datetime, timedelta
import collections

class NLPCleanScraper:
    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://www.baidu.com/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'cross-site',
            'Sec-Fetch-User': '?1',
            # 如果仍然遇到403，请尝试在浏览器中登录百度百科，按F12复制Cookie填入下方
            'Cookie': 'BAIDUID_BFESS=CD951A3B6816966F7134EE861663C127:FG=1; zhishiTopicRequestTime=1766414751119; ppfuid=FOCoIC3q5fKa8fgJnwzbE67EJ49BGJeplOzf+4l4EOvDuu2RXBRv6R3A1AZMa49I27C0gDDLrJyxcIIeAeEhD8JYsoLTpBiaCXhLqvzbzmvy3SeAW17tKgNq/Xx+RgOdb8TWCFe62MVrDTY6lMf2GrfqL8c87KLF2qFER3obJGmwKsggvQ93gWKRRjAgI2cnGEimjy3MrXEpSuItnI4KDx+TfKYZKPWGqI4sgP0w/90v1Q6wbZ/cnN+cvSCK1Ndlm0+MQtbQyIO4UWGTo2NHAgwOj0D7V9Fosz5XirWhTWHh9EHAWDOg2T1ejpq0s2eFy9ar/j566XqWDobGoNNfmfpaEhZpob9le2b5QIEdiQe/UUOOs2fkuUdvUcIGr3f6FCMUN0p4SXVVUMsKNJv2T5I/Spk0RWuhYOetaf5ccwsjWEdP4XOg/wE7MjKnKsD7zkWOdjFiy1eD/0R8HcRWYoPuRNpHVOaR4Ef46MDFxMap+HBavJhpxl858h16cMtKQmxzisHOxsE/KMoDNYYE7ucLE22Bi0Ojbor7y6SXfVj7+B4iuZO+f7FUDWABtt/WWQqHKVfXMaw5WUmKnfSR5wwQa+N01amx6X+p+x97kkGmoNOSwxWgGvuezNFuiJQdt51yrWaL9Re9fZveXFsIu/gzGjL50VLcWv2NICayyI8BE9m62pdBPySuv4pVqQ9Sl1uTC//wIcO7QL9nm+0N6JgtCkSAWOZCh7Lr0XP6QztjlyD3bkwYJ4FTiNanaDaDIgLynzyQbbJnP4EF8rjhkrT5GqTqlQhFY4MMdtQu+OowWXBGubEJ7Vj8FXKXP6ukSGk66HDxtjKMU4HPNa0dthF7UsHf7NW9eE+gwuTQSa7GLWfOy9+ap4iFBQsmjpefgOF89jAHLbnVUejtrqqvdVSQ/4gzJOb0DGzeEZ5GeyMzgLkehXgk0UZz/MyefUOQXlV3f0HZXSpuSxTnDK9hXLZEuBHhU0MbbED5DF65/h/gBRkDPTwtXjYtgzmW74m0fDU2MZaxpBZZF8YurfocYcmDdcxFKeoIFQmVqAoAU+3YcXQt2xKThZZyV1v3sCvnzidUZtKM9cRRUfRWBtQSb50APM+gs/408xg7KHCB8AOKpZpfIpPhQ0RJhew8GR0aTqYsJo1IRCwM3UbbrvtJ7eqPMNzJcGcSYcQWm1FubInMonve94c+p8Vi2wc72MfReeFiTzMp1G6pDt2e40gPDGbdQI+jba4UjRlyA+9CbTXeo25gLlwEeBBlbK/gwjLfyky6FPenbJgt/vQK9TAiTA==; XFI=f927f140-df44-11f0-9a69-6bc52427cebd; XFCS=BCB378A8584EE2D14DC972CEB8BFD2BDC42834E6688AF45A0FAE3307452D16F1; XFT=hfyaSwlxvQ1IC3hHbezbZAnWntI33xpaKkFdWpgalxU=; BDUSS=n5uTHBRNnV-dWpXRDFLaWx2SXNsQXQ5bU0wa3VUd1NhYkw5SXJjRUJsZlQ1bkJwRUFBQUFBJCQAAAAAAAAAAAEAAACftbjZQ2FydGVyWWluMjMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANNZSWnTWUlpb; BDUSS_BFESS=n5uTHBRNnV-dWpXRDFLaWx2SXNsQXQ5bU0wa3VUd1NhYkw5SXJjRUJsZlQ1bkJwRUFBQUFBJCQAAAAAAAAAAAEAAACftbjZQ2FydGVyWWluMjMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANNZSWnTWUlpb; channel=passport.baidu.com; baikeVisitId=8b314d8f-d51d-4e0b-aeef-04eae8fbce5c; BKWPF=3; ab_sr=1.0.1_OGRhYjUyYTU2YzM3ZGIzYTJkZjEzM2I4Zjk3ZWViZGRjY2JjYmY3YzAxYTA3YWMwY2MzYzQ0YWNiMTAyZjdlOGY1YWNkNDVlOWIxN2U1N2JhNWRjZWEwODhjZjQwYzI3YTM4YTE1ZmIyNzhmZDk2NWQ2YjcwNzJhNmMzZWU3OGIzZDNhYWE2YmZmYjgzZWQ2YjlmNWUzNDE3ZmZlNDNjNDBkYmYzYTg2NjNiZGVhYmZlM2VlODVlM2Q0ZWFiYjQx'
        }
        self.session.headers.update(self.headers)
        self.history_set = set() # 用于内存去重
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

    def clean_text(self, text):
        """
        核心清洗函数：
        1. 剔除 HTML 实体 (如 &nbsp;)
        2. 仅保留汉字、英文字母、数字、常用中文标点
        3. 去除多余空格
        """
        # 替换常见的 HTML 实体
        text = re.sub(r'&[a-z]+;|&#\d+;', '', text)
        # 正则表达式说明：保留汉字(\u4e00-\u9fa5)、数字、字母、及基础标点(，。！？：)
        # 排除所有特殊的控制字符、广告乱码
        clean_pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？：、“”]')
        text = clean_pattern.sub('', text)
        return text.strip()

    def is_valid(self, text):
        """判断语料是否合格"""
        # 长度太短（小于8个字）通常是广告片段、作者名或日期，不适合做NLP语料
        if len(text) < 8:
            return False
        # 去重检查
        if text in self.history_set:
            return False
        # 排除包含特定广告关键词的行
        ads_keywords = ['点击查看', '扫码', '关注微信', '下载APP', '详情咨询']
        if any(kw in text for kw in ads_keywords):
            return False
        return True

    def _save_line(self, text, filename):
        file_path = os.path.join(self.storage_path, filename)
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(text + "\n")
        self.history_set.add(text)

    def crawl_news(self, total_target, start_date_str):
        print(f"\n[开始爬取新闻] 目标: {total_target}条清洗后的语料")
        current_date = datetime.strptime(start_date_str, "%Y%m%d")
        count = 0
        
        while count < total_target:
            # 改用中国新闻网滚动新闻，因为新浪新闻的旧链接失效了
            # 格式: https://www.chinanews.com.cn/scroll-news/2024/0101/news.shtml
            date_path = current_date.strftime("%Y/%m%d")
            date_disp = current_date.strftime("%Y%m%d")
            url = f"https://www.chinanews.com.cn/scroll-news/{date_path}/news.shtml"
            
            try:
                res = self.session.get(url, timeout=10)
                res.encoding = 'utf-8'
                if res.status_code == 200:
                    soup = BeautifulSoup(res.text, 'html.parser')
                    # 修改选择器以匹配中国新闻网结构 (.content_list 下的 .dd_bt 类中的 a 标签)
                    links = soup.select('.content_list .dd_bt a')
                    for l in links:
                        raw_text = l.get_text()
                        cleaned = self.clean_text(raw_text)
                        
                        if self.is_valid(cleaned) and count < total_target:
                            self._save_line(cleaned, "corpus_news_clean.txt")
                            count += 1
                else:
                    print(f"日期 {date_disp} 页面未找到或请求失败 (Status: {res.status_code})")
                
                print(f"日期 {date_disp} 进度: {count}/{total_target}")
                current_date -= timedelta(days=1)
                time.sleep(random.uniform(1.2, 2.8)) # 模拟人类阅读
                
            except Exception as e:
                print(f"连接错误: {e}")
                time.sleep(10) # 遇错多休息一会儿
        return count

    def crawl_social(self, total_target):
        print(f"\n[开始爬取社交语料] 目标: {total_target}条清洗后的语料")
        count = 0
        for i in range(0, 500): # 翻页上限
            if count >= total_target: break
            
            start = i * 25
            url = f"https://www.douban.com/group/explore?start={start}"
            try:
                res = self.session.get(url, timeout=10)
                if res.status_code == 200:
                    soup = BeautifulSoup(res.text, 'html.parser')
                    titles = soup.select('.bd h3 a')
                    for t in titles:
                        cleaned = self.clean_text(t.get_text())
                        if self.is_valid(cleaned) and count < total_target:
                            self._save_line(cleaned, "corpus_social_clean.txt")
                            count += 1
                    print(f"豆瓣翻页第 {i+1} 页，当前有效总数: {count}")
                    time.sleep(random.uniform(2.0, 4.0)) # 豆瓣抓取频率需更低
                else:
                    print(f"豆瓣请求失败，状态码: {res.status_code}。可能触发了反爬限制。")
                    break
            except Exception as e:
                print(f"豆瓣爬取发生错误: {e}")
                break
        return count

    def crawl_baike(self, total_target):
        print(f"\n[开始爬取百度百科] 目标: {total_target}条清洗后的语料")
        # 初始种子链接
        seeds = ['人工智能', '自然语言处理', '深度学习', '机器学习', '计算机科学', 'Python', 'Java', '历史', '文学', '艺术']
        queue = collections.deque([f"https://baike.baidu.com/item/{k}" for k in seeds])
        count = 0
        
        # 记录已访问的URL，避免重复爬取
        visited_urls = set()

        while count < total_target and queue:
            url = queue.popleft()
            if url in visited_urls:
                continue
            visited_urls.add(url)
            
            try:
                # 随机延时，模拟人类行为
                time.sleep(random.uniform(1.0, 3.0))
                res = self.session.get(url, timeout=10)
                if res.status_code == 200:
                    res.encoding = 'utf-8'
                    soup = BeautifulSoup(res.text, 'html.parser')
                    
                    # 提取摘要 (.lemma-summary 或 .J-summary)
                    summary_div = soup.find('div', class_='lemma-summary') or soup.find('div', class_='J-summary')
                    if summary_div:
                        raw_text = summary_div.get_text()
                        cleaned = self.clean_text(raw_text)
                        if self.is_valid(cleaned):
                            self._save_line(cleaned, "corpus_baike_clean.txt")
                            count += 1
                            print(f"百科抓取进度: {count}/{total_target} - {url}")
                    
                    # 提取新链接加入队列
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        # 简单的链接过滤，只保留 /item/ 开头的链接
                        if href.startswith('/item/') and '?' not in href:
                            full_url = f"https://baike.baidu.com{href}"
                            if full_url not in visited_urls:
                                queue.append(full_url)
                                
                    time.sleep(random.uniform(1.0, 2.0))
                elif res.status_code == 403:
                    print(f"访问被拒绝 (403): {url} - 百度百科反爬较严，可能需要更复杂的Headers或Cookies")
                    time.sleep(2)
                else:
                    print(f"请求失败 ({res.status_code}): {url}")

            except Exception as e:
                print(f"百科爬取错误: {e}")
                
        return count

def main():
    save_dir = input("存储位置 (默认 ./clean_data): ") or "clean_data"
    data_size = int(input("每种语料需要多少条有效数据 (如 10000): ") or 100)
    
    print("\n请选择要爬取的语料源:")
    print("1. 新闻 (News)")
    print("2. 社交媒体 (Douban)")
    print("3. 百度百科 (Baike)")
    print("4. 全部 (All)")
    choice = input("请输入选项 (1-4, 默认4): ") or "4"

    scraper = NLPCleanScraper(save_dir)

    if choice in ['1', '4']:
        start_date = input("新闻起始日期 (如 20250520): ") or "20250520"
        scraper.crawl_news(data_size, start_date)
        scraper.history_set.clear()

    if choice in ['2', '4']:
        scraper.crawl_social(data_size)
        scraper.history_set.clear()

    if choice in ['3', '4']:
        print("\n注意: 百度百科反爬严格，如果遇到403错误，请尝试在代码 headers 中添加 Cookie。")
        scraper.crawl_baike(data_size)

if __name__ == "__main__":
    main()