#!/usr/bin/env python3

import glob
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

        try:
            doc = json.load(open(fn, 'r'))
        except json.JSONDecodeError as exc:
            click.echo('No dice on scanning {}: {}'
                       .format(fn, exc))
            continue

        for checkin in doc:
            yield checkin['checkin']['comments']


class SentenceGenerator(object):
    def __init__(self, nlp, filenames):
        self.nlp = nlp
        self.sentences = file_comment_collector(filenames)

    def __iter__(self):
        for i, doc in enumerate(self.nlp.pipe(self.sentences,
                                              batch_size=50,
                                              n_threads=4)):
            for sentence in doc.sents:
                try:
                    yield [t.text for t in sentence]
                except Exception as exc:
                    click.echo('Sentence unusable: {}'.format(exc))


@click.group()
@click.pass_context
def cli(ctx): pass


@cli.command('generate-model')
@click.option('-i', 'data_path',
              type=click.Path(exists=True, file_okay=False))
@click.option('-o', 'model_path', type=click.Path(file_okay=True,
                                                  dir_okay=False,
                                                  writable=True))
@click.pass_context
def generate_model(ctx, data_path, model_path):
    # find all input files
    fns = glob.glob(os.path.join(data_path, 'brewery-[0-9].json'))

    click.echo('Building model')

    nlp = spacy.load('en')

    # build, write model
    model = gensim.models.Word2Vec(min_count=2, workers=4)
    model.build_vocab(SentenceGenerator(nlp, fns))
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
    click.echo('Starting app')
    cli(obj={})
