#!/usr/bin/env python3

import csv
import glob
import gzip
import json
import os
import pprint

import click

import gensim

from numpy import array as np_array

import spacy


def file_comment_collector(filenames):
    for fn in filenames:
        click.echo('Reading comment(s) from {}'.format(fn))

        with gzip.open(fn, 'rt') as fh:
            try:
                doc = json.load(fh)
            except json.JSONDecodeError as exc:
                click.echo('No dice on scanning {}: {}'
                           .format(fn, exc))
                continue

            for checkin in doc:
                style_token = checkin['beer']['style'].replace(' ', '_')
                yield (style_token, checkin['checkin']['comments'])


class SentenceGenerator(object):
    def __init__(self, fh):
        """Iterates sentences from a CSV file

        Arguments:
            filename: Path to CSV file of <label, sentence> rows
        """

        self.sentences = csv.reader(fh)

    def __iter__(self):
        for row in self.sentences:
            words = [w.strip().lower() for w in row[1].split()]

            try:
                yield gensim.models.doc2vec.LabeledSentence(words, [row[0]])
            except Exception as exc:
                    click.echo('Sentence unusable: {}'.format(exc))


@click.group()
@click.pass_context
def cli(ctx): pass


@cli.command('generate-sentences')
@click.option('-i', 'data_path',
              type=click.Path(exists=True, file_okay=False))
@click.option('-o', 'output_fh', type=click.File('a'))
@click.pass_context
def generate_sentences(ctx, data_path, output_fh):
    fns = glob.glob(os.path.join(data_path, '*.json.gz'))
    writer = csv.writer(output_fh)

    nlp = spacy.load('en')

    for token, comment in file_comment_collector(fns):
        doc = nlp(comment)

        for sentence in doc.sents:
            words = [
                t.text for t in sentence
                if not
                    t.is_stop
                    and not t.is_punct
                    and not t.is_digit
            ]
            writer.writerow([token, ' '.join(words).strip()])


@cli.command('generate-model')
@click.option('-i', 'data_path', type=click.File('r'))
@click.option('-o', 'model_path', type=click.Path(file_okay=True,
                                                  dir_okay=False,
                                                  writable=True))
@click.option('--epochs', 'epochs', type=click.INT, default=10)
@click.pass_context
def generate_model(ctx, data_path, model_path, epochs):
    # build, write model
    model = gensim.models.Doc2Vec(min_count=5, workers=4,
                                  alpha=0.025, min_alpha=0.025)

    click.echo('Building model vocab')

    model.build_vocab(SentenceGenerator(data_path))

    # train model
    for e in range(epochs):
        click.echo('Training epoch {}'.format(e + 1))
        data_path.seek(0)
        model.train(SentenceGenerator(data_path))
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    model.init_sims(replace=True)
    model.save(model_path)

    click.echo('Model built: {}'.format(model_path))


@cli.command('show-vocab')
@click.option('-m', 'model_path', type=click.Path(exists=True, dir_okay=False),
              required=True)
@click.pass_context
def get_similarity(ctx, model_path):
    model = gensim.models.Word2Vec.load(model_path)
    model.init_sims(replace=True)

    pprint.pprint(model.vocab)


@cli.command('similarity')
@click.option('-m', 'model_path', type=click.Path(exists=True, dir_okay=False),
              required=True)
@click.option('-t', 'term', required=True)
@click.pass_context
def get_similarity(ctx, model_path, term):
    model = gensim.models.Word2Vec.load(model_path)
    model.init_sims(replace=True)

    sim = model.most_similar(term)
    print(sim)


if __name__ == '__main__':
    assert gensim.models.word2vec.FAST_VERSION > -1
    cli(obj={})
