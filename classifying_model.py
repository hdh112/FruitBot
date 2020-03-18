# Reference: 	https://www.tensorflow.org/tutorials/load_data/text

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

DIRECTORY = 'words/'
FILE_NAMES = ['[words]fruit_order.txt']
labeled_data_sets = []
vocabulary_set = set()

# TODO: make labels more specific, according to tags and intent
# Label 0: order
def labeler(sentence, index):
	return sentence, tf.cast(index, tf.int64)

#######################################################
# Label data and update vocabulary
for i, file_name in enumerate(FILE_NAMES):
	lines_dataset = tf.data.TextLineDataset(DIRECTORY + file_name)
	# TODO: iterate lines_dataset only once;
	# 		for now, iterates twice because of decoding bytes issue
	labeled_dataset = lines_dataset.map(lambda sent: labeler(sent, i))
	for line in lines_dataset.as_numpy_iterator():
		# TODO: handle multiple meanings of a single word
		vocabulary_set.update(line.rstrip().split())
			
	labeled_data_sets.append(labeled_dataset)

#######################################################
# Concatenate labeled data and shuffle
all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
	all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

# TODO: make dataset larger
BUFFER_SIZE = 30
BATCH_SIZE = 10
TAKE_SIZE = 3

all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

#######################################################
# Encode vocabulary
# TODO: encode punctuation marks
print(len(vocabulary_set))
print(vocabulary_set)
print(next(iter(all_labeled_data))[0])
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

def encode(text_tensor, label):
	encoded_text = encoder.encode(text_tensor.numpy())
	return encoded_text, label


def encode_map_fn(text, label):
	encoded_text, label = tf.py_function(encode,
										 inp=[text, label],
										 Tout=(tf.int64, tf.int64))

	# Adjust shape for components of `tf.data.Dataset`
	encoded_text.set_shape([None])
	label.set_shape([])

	return encoded_text, label

all_encoded_data = all_labeled_data.map(encode_map_fn)

#######################################################
# Split dataset into train and test batches
train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE)

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE)