import os
import random
import re
import sys
from random import choice, choices

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
    results = {page : 0 for page in corpus.keys()}

    # No link on page
    if len(corpus[page]) == 0:
        for link in corpus.keys():
            results[link] = results[link] + (1 / len(corpus))
        return results
    
    # Random link with probability of damping 
    for link in corpus[page]:
        results[link] = results[link] + damping_factor * (1 / len(corpus[page]))

    # Random link with probability of (1 - damping) 
    for link in corpus.keys():
        results[link] = results[link] + (1 - damping_factor) * (1 / len(corpus))

    return results


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ranking = {key: 0 for key in corpus.keys()}
    pages = [page for page in corpus.keys()]
    page = choice(pages)
    ranking[page] = ranking[page] + 1

    tramo = transition_model(corpus, page, damping_factor)

    # Generating remaining samples
    count = 2 
    while count <= n:
        population = [key for key in tramo.keys()]
        weights = [value for value in tramo.values()]
        page = choices(population, weights, k=1)[0]
        ranking[page] = ranking[page] + 1
        tramo = transition_model(corpus, page, damping_factor)
        count += 1

    # Normalization
    for page in ranking.keys():
        ranking[page] = ranking[page] / n

    return ranking

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

    # Placing links to every page on pages without any links
    new_corpus = dict()
    for key in corpus.keys():
        val = corpus[key] if corpus[key] != set() else set(corpus.keys())
        new_corpus[key] = val

    while True:
        current_rank = page_ranks
        new_ranking = {key: 0 for key in new_corpus.keys()}

        for page in corpus.keys():
            choice_a = 1 / N
            choice_b = sum([page_ranks[i] / len(corpus[i]) for i in corpus.keys() if page in corpus[i]])

            new_ranking[page] = (1 - damping_factor) * choice_a + damping_factor * choice_b

        page_ranks = new_ranking

        differences = [i - j for i, j in zip(current_rank.values(), new_ranking.values())]

        if all(difference <= 0.001 for difference in differences):
            break

    return page_ranks


if __name__ == "__main__":
    main()
