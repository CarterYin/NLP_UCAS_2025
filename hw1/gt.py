"""面向环球时报英文网主页的轻量级爬虫。

该脚本从 https://www.globaltimes.cn/index.html 出发，遍历域内链接，持续收集页面文本，
并在累计文本体积达到约 2MB、5MB、10MB 时，分别将快照写入 data/ 目录。

依赖：
	pip install requests beautifulsoup4
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from typing import Deque, Iterable, List, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.globaltimes.cn/index.html"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SESSION = requests.Session()
SESSION.headers.update(
	{
		"User-Agent": (
			"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
			"(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
		),
		"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
		"Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
	}
)

MAX_PAGES = 400
TIMEOUT = 10
REQUEST_DELAY = 0.75
THRESHOLDS = [2 * 1024 * 1024, 5 * 1024 * 1024, 10 * 1024 * 1024]
ELLIPSIS_SUFFIXES = ("...", "…")
MAX_DEPTH = 2

NAVIGATION_TERMS = {
	"CHINA",
	"Politics",
	"Society",
	"Diplomacy",
	"Military",
	"Science",
	"Odd",
	"China Graphic",
	"SOURCE",
	"GT Voice",
	"Insight",
	"Economy",
	"Comments",
	"Company",
	"B&R Initiative",
	"Biz Graphic",
	"OPINION",
	"Editorial",
	"Observer",
	"Global Minds",
	"Asian Review",
	"Top Talk",
	"Viewpoint",
	"Columnists",
	"Cartoon",
	"LIFE",
	"Cultural Influencer",
	"Attitudes",
	"Culture",
	"Entertainment",
	"Travel",
	"VISUAL NEWS",
	"Video",
	"Graphics",
	"Gallery",
	"Specials",
	"MISCELLANEOUS",
	"In-Depth",
	"World",
	"Sport",
	"Newsletter",
	"About Us",
	"Careers",
	"Contact Us",
	"Advertisement",
	"Terms of Service",
	"MORE",
	"RELATED ARTICLES",
}

FOOTER_PATTERNS = (
	r"^By\s+.+\|\s+\d{4}/\d{1,2}/\d{1,2}",
	r"All Right Reserved",
	r"^Follow us",
	r"^Most Read$",
	r"^Hot on",
)


def fix_mojibake(text: str) -> str:

	if any(marker in text for marker in ("â", "Ã", "Â")):
		try:
			candidate = text.encode("latin-1", errors="ignore").decode(
				"utf-8", errors="ignore"
			)
			if candidate:
				text = candidate
		except UnicodeDecodeError:
			pass
	return text.replace("\ufeff", "").replace("\u00a0", " ")


def contains_cjk(text: str) -> bool:
	return bool(re.search(r"[\u4e00-\u9fff]", text))


def should_skip_line(line: str) -> bool:
	if not line:
		return True
	if line in NAVIGATION_TERMS:
		return True
	if contains_cjk(line):
		return True
	if any(line.endswith(suffix) for suffix in ELLIPSIS_SUFFIXES) and len(line) < 140:
		return True
	if line.upper() == line and len(line) <= 40:
		return True
	for pattern in FOOTER_PATTERNS:
		if re.search(pattern, line, flags=re.IGNORECASE):
			return True
	return False


def normalize_line(text: str) -> str | None:
	if not text:
		return None
	text = fix_mojibake(text)
	text = re.sub(r"\s+", " ", text).strip()
	if not text or should_skip_line(text):
		return None
	return text


def clean_text(text: str) -> str:
	lines: List[str] = []
	for raw_line in text.splitlines():
		line = normalize_line(raw_line)
		if line:
			lines.append(line)
	return "\n".join(lines)


def is_in_domain(url: str) -> bool:
	parsed = urlparse(url)
	if parsed.scheme not in {"http", "https"}:
		return False
	host = parsed.netloc.lower()
	return host.endswith("globaltimes.cn")


def normalize_link(base_url: str, link: str) -> str | None:
	if not link:
		return None
	absolute = urljoin(base_url, link)
	if not is_in_domain(absolute):
		return None
	parsed = urlparse(absolute)
	cleaned = parsed._replace(fragment="").geturl()
	return cleaned


def fetch(url: str) -> str | None:
	try:
		response = SESSION.get(url, timeout=TIMEOUT)
	except requests.RequestException as exc:
		print(f"请求失败: {url} | 错误: {exc}")
		return None

	if response.status_code >= 400:
		print(f"访问失败: {url} | 状态码: {response.status_code}")
		return None
	if not response.encoding or response.encoding.lower() == "iso-8859-1":
		apparent = response.apparent_encoding or "utf-8"
		response.encoding = apparent
	else:
		response.encoding = response.encoding
	return response.text


def extract_article_text(html: str, url: str) -> str | None:
	if url == BASE_URL:
		return None
	soup = BeautifulSoup(html, "html.parser")
	for tag in soup(["script", "style", "noscript", "template"]):
		tag.decompose()

	title: str | None = None
	meta_title = soup.find("meta", attrs={"property": "og:title"})
	if meta_title and meta_title.get("content"):
		title = normalize_line(meta_title.get("content", ""))
	if not title:
		heading = soup.find("h1")
		if heading:
			title = normalize_line(heading.get_text(" ", strip=True))
	if not title:
		doc_title = soup.find("title")
		if doc_title and doc_title.string:
			title = normalize_line(doc_title.string)

	containers: List = []
	seen_container_ids: Set[int] = set()

	def register_container(tag) -> None:
		if tag and id(tag) not in seen_container_ids:
			seen_container_ids.add(id(tag))
			containers.append(tag)

	register_container(soup.find("article"))
	for candidate in soup.select("div.article-content"):
		register_container(candidate)
	for candidate in soup.select("div#articleContent"):
		register_container(candidate)
	for candidate in soup.select("section.article"):
		register_container(candidate)
	for candidate in soup.find_all(["div", "section"], class_=True):
		class_name = " ".join(candidate.get("class", []))
		if re.search(r"(article|content)", class_name, flags=re.IGNORECASE):
			register_container(candidate)

	paragraphs: List[str] = []
	for container in containers:
		for para in container.find_all("p"):
			text = normalize_line(para.get_text(" ", strip=True))
			if text and text not in paragraphs:
				paragraphs.append(text)
	if not paragraphs:
		for para in soup.find_all("p"):
			text = normalize_line(para.get_text(" ", strip=True))
			if text and text not in paragraphs:
				paragraphs.append(text)

	if not paragraphs:
		return None

	if not containers:
		total_length = sum(len(p) for p in paragraphs)
		if total_length < 400:
			return None

	body = "\n\n".join(paragraphs)
	if title:
		return f"{title}\n\n{body}"
	return body


def extract_links(html: str, base_url: str) -> Iterable[str]:
	soup = BeautifulSoup(html, "html.parser")
	for anchor in soup.find_all("a"):
		href = anchor.get("href")
		normalized = normalize_link(base_url, href)
		if normalized:
			yield normalized


def save_snapshot(buffer: bytearray, threshold: int) -> None:
	size_mb = threshold // (1024 * 1024)
	filename = f"globaltimes_snapshot_{size_mb}MB.txt"
	path = os.path.join(DATA_DIR, filename)
	os.makedirs(DATA_DIR, exist_ok=True)
	with open(path, "wb") as file:
		file.write(buffer[:threshold])
	print(f"已写入快照: {filename}")



def crawl() -> None:
	queue: Deque[tuple[str, int]] = deque([(BASE_URL, 0)])
	visited: Set[str] = set()
	buffer = bytearray()
	remaining_thresholds = THRESHOLDS.copy()

	print("开始抓取环球时报英文网相关链接...\n")

	while queue and remaining_thresholds and len(visited) < MAX_PAGES:
		url, depth = queue.popleft()
		if url in visited:
			continue
		visited.add(url)

		html = fetch(url)
		if not html:
			continue

		text = extract_article_text(html, url)
		if text:
			encoded = text.encode("utf-8") + b"\n\n"
			buffer.extend(encoded)
			print(
				f"累计抓取大小: {len(buffer) / (1024 * 1024):.2f} MB | 已访问页面: {len(visited)}"
			)

		if depth < MAX_DEPTH:
			for link in extract_links(html, url):
				if link not in visited:
					queue.append((link, depth + 1))

		while remaining_thresholds and len(buffer) >= remaining_thresholds[0]:
			threshold = remaining_thresholds.pop(0)
			save_snapshot(buffer, threshold)

		time.sleep(REQUEST_DELAY)


if __name__ == "__main__":
	crawl()
