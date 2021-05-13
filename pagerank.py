import urllib.error
import urllib.parse
import urllib.request
from fractions import Fraction

import numpy as np
from bs4 import BeautifulSoup


def create_l(url_list):
    n = len(url_list)
    l = np.zeros((n, n))

    # Create matrix l (Google matrix) with the given urls
    for url in url_list:
        url_idx = url_list.index(url)
        count = 0

        soup = BeautifulSoup(urllib.request.urlopen(url).read(), 'html.parser')
        links = soup('a')

        for link in links:
            link_text = str(link.get('href', None))
            for pattern in url_list:
                if pattern != url and (pattern == link_text or link_text == pattern + "/"):
                    if l[url_list.index(pattern)][url_idx] == 0:
                        count += 1
                    l[url_list.index(pattern)][url_idx] = 1

        if count > 0:
            l[:, url_idx] = l[:, url_idx] / count

    # Handle dangling pages
    for i in range(n):
        if np.all(l[:, i] == 0):
            l[:, i] = np.full(n, Fraction(1, n))

    return l


def pagerank(l, d, iterations):
    n = l.shape[1]
    p = np.random.rand(n, 1)  # Start with random values for the pageranks ...
    p = p / np.linalg.norm(p, 1)  # ... normalized
    l_ext = (d * l + (1 - d) / n)
    for i in range(iterations):
        p = l_ext @ p
    return p


if __name__ == '__main__':
    urls = ["https://www.informatik.uni-rostock.de",
            "https://www.facebook.com/universitaet.rostock",
            "https://lsf.uni-rostock.de/vvz",
            "https://studip.uni-rostock.de",
            "https://www.ief.uni-rostock.de",
            "https://www.uni-rostock.de"]

    pagerank = pagerank(create_l(urls), 0.85, 200)
    print(pagerank)
