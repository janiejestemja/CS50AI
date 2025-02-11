import os
import random
import re
import sys
from random import choice, choices
import numpy as np
import pandas as pd

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    random_factor = (1 - DAMPING) / len(corpus)
    results = {page : random_factor for page in corpus.keys()}

    if len(corpus[page]) == 0:
        return results

    target_factor = DAMPING / len(corpus[page])
    for link in corpus[page]:
        results[link] += target_factor

    return results


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = [page for page in corpus.keys()]
    start_page = choice(pages)


    visited_pages = []
    for i in range(n):

        tramo = transition_model(corpus, start_page, damping_factor)

        probabilities = tramo.values()
        next_page = choices(pages, probabilities)

        visited_pages.append(next_page)

    df = pd.DataFrame(visited_pages).astype({0 : "category"})

    uniques = df[0].unique()

    variable = df[0].value_counts()

    page_ranks = {uniques[i] : int(variable[i]) for i in range(len(df[0].unique()))}

    for page in page_ranks:
        page_ranks[page] = page_ranks[page] / n

    return page_ranks

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)

    page_ranks = {page : (1 / N) for page in corpus.keys()}


    last_change = 1
    while last_change > 0.0001:

        for page in page_ranks.keys():

            rank_over_link = []

            for link in corpus:
                match len(corpus[link]):
                    case 0:
                        rank_over_link.append(page_ranks[link] / N)
                    case _:
                        if page in corpus[link]:
                            rank_over_link.append(page_ranks[link] / len(corpus[link]))


            page_ranks[page] = ((1 - damping_factor) / N ) + damping_factor * sum(rank_over_link)

        last_change = 1 - sum(page_ranks.values())

    return page_ranks


if __name__ == "__main__":
    main()
