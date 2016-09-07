# beer-to-vec

Uses word2vec to analyze beer comments on Beer Advocate.
Input data is taken from data scraped by `grist`.

## Installation

Clone this repo into a fresh Python 3.5+ virtual environment,
then install its requirements:

```bash
$ pip install -r requirements.txt
```

If this is your first time setting up, also install the Spacy data models:

```bash
$ sputnik --name spacy --repository-url http://index.spacy.io install en==1.1.0
```

Note: model installation downloads approximately 500 mb of data. Don't do this on your cell connection.

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
