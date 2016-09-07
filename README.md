# beer-to-vec

Uses word2vec to analyze beer comments on Beer Advocate.
Input data is taken from data scraped by `grist`.

## Installation

Clone this repo into a fresh Python 3.5+ virtual environment,
then install its requirements:

```bash
$ pip install -r requirements.txt
```

## Usage

First, train the model:

```bash
$ ./app.py generate-model -i /path/to/*.json -o trained-model
```

Optionally, explore the vocabulary this generates:

```bash
$ ./app.py show-vocab -m /path/to/trained-model
```

Then, explore the vocab:

```bash
$ ./app.py similarity -t <term> -m /path/to/trained-model
```

e.g.,

```bash
$ ./app.py similarity -t dank -m ./trained-model
```
