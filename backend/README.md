# To Be Updated

# EPFL Human Relations and PolyLex Scraper

This repository contains tools for scraping and processing documents from the EPFL Human Relations website and PolyLex resources. It automates the retrieval of documents, conversion to PDF, and subsequent text processing for embedding and indexing.

## Overview

The scraping targets two main sources:
- **Human Relations Site at EPFL**: Contains various human resources documents.
- **LEXs of PolyLex**: Includes legislative texts relevant to the Polytechnic context.

Documents from these sources are first converted into PDF format. Each PDF is then divided into what are referred to as `big_chunks`. Each `big_chunk` includes:
- **Page Content**: The text of a single page, used for embedding and indexing.
- **Three_Page_Content**: Concatenation of the previous page, current page, and the next page (where applicable). This concatenated text is utilized in the Retrieval-Augmented Generation (RAG) process with Large Language Models (LLMs).

Further, the page content is segmented into `small_chunks` of 500 tokens. These tokens are embedded in French and translated into English, with the English content also being embedded for dual-language support.

## Setup and Installation

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt

cd epfl_hr_scraper
scrapy crawl hr_spider # Scrapes documents to data/HR/HR.db
scrapy crawl lex_spider # Scrapes LEXs to data/LEXs/LEXs.db

cd ..


```