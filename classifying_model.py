# Reference: 	https://www.tensorflow.org/tutorials/load_data/text

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

DIRECTORY = 'words/'
FILE_NAMES = ['[words]fruit_order.txt']
vocabulary_set = set()

# TODO: make labels more specific, according to tags and intent
# Label 0: order
def labeler(sentence, index):
	return sentence, tf.cast(index, tf.int64)

labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
	lines_dataset = tf.data.TextLineDataset(DIRECTORY + file_name)
	# TODO: iterate lines_dataset only once;
	# 		for now, iterates twice because of numpy-tensorflow compatability issue
	labeled_dataset = lines_dataset.map(lambda sent: labeler(sent, i))
	for line in lines_dataset.as_numpy_iterator():
		# TODO: handle multiple meanings of a single word
		vocabulary_set.update(line.rstrip().split())
			
	labeled_data_sets.append(labeled_dataset)

print(len(vocabulary_set))
print(vocabulary_set)
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
	all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

# TODO: make dataset larger
BUFFER_SIZE = 200
BATCH_SIZE = 16
TAKE_SIZE = 20

# all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
# all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)