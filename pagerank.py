import urllib.request, urllib.parse, urllib.error
from fractions import Fraction
from bs4 import BeautifulSoup
import numpy as np
import re


def createL(urls):
    n = len(urls)
    l = np.zeros((n, n))

    # Create matrix l (Google matrix) with the given urls
    for url in urls:
        urlIdx = urls.index(url)
        count = 0

        soup = BeautifulSoup(urllib.request.urlopen(url).read(), 'html.parser')
        links = soup('a')

        for link in links:
            linkText = str(link.get('href', None))
            for pattern in urls:
                if (pattern != url and (pattern == linkText or linkText == pattern + "/")):
                    if l[urls.index(pattern)][urlIdx] == 0:
                        count += 1
                    l[urls.index(pattern)][urlIdx] = 1
        
        if count > 0:
           l[:,urlIdx] = l[:,urlIdx] / count

    # Handle dangling pages
    for i in range(n):
        if np.all(l[:,i] == 0):
            l[:,i] = np.full(n, Fraction(1, n))
    
    return l

def pagerank(l, d, iterations):
    n = l.shape[1]
    p = np.random.rand(n, 1) # Start with random values for the pageranks ...
    p = p / np.linalg.norm(p, 1) # ... normalized 
    lExt = (d * l + (1 - d) / n)
    for i in range(iterations):
        p = lExt @ p
    return p

urls = ["https://www.informatik.uni-rostock.de",
        "https://www.facebook.com/universitaet.rostock",
        "https://lsf.uni-rostock.de/vvz",
        "https://studip.uni-rostock.de",
        "https://www.ief.uni-rostock.de",
        "https://www.uni-rostock.de"]

p = pagerank(createL(urls), 0.85, 200)
print(p)