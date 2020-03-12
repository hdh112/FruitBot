# Reference: https://www.tensorflow.org/tutorials/load_data/text

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
# import os

DIRECTORY = 'chunks/'
FILE_NAMES = ['[chunk]fruit_order.txt']
vocabulary_set = set()

# TODO: make labels more specific, according to tags and intent
# Label 0: order
def labeler(sentence, index):
	untagged_words = tagged_word.split('/')[0] \
						for tagged_word in sentence.rstrip().split()
	vocabulary_set.update(untagged_words)

	return " ".join(untagged_words), tf.cast(index, tf.int64)

labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
	# lines_dataset = tf.data.TextLineDataset(tf.cast(DIRECTORY + file_name, tf.string))
	lines_dataset = tf.data.TextLineDataset(DIRECTORY + file_name)
	labeled_dataset = lines_dataset.map(lambda stnc: labeler(stnc, i))
	labeled_data_sets.append(labeled_dataset)

# print(len(vocabulary_set))
# TODO: handle regex "\W+" that comes along with Korean(non-alphanumeric) characters
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

# TODO: make dataset larger
BUFFER_SIZE = 200
BATCH_SIZE = 16
TAKE_SIZE = 20

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
	all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

# all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

