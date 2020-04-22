# Reference: 	https://www.tensorflow.org/tutorials/load_data/text

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import tensorflow.keras.layers as layers

# DIRECTORY = 'words/'
# FILE_NAMES = ['[words]fruit_order.txt']
DIRECTORY = 'essential_phrases/'
FILE_NAMES = ['[essential]fruit_order.txt']

# TODO: move sentences with multiple intents to a new .txt
FILE_TAGS = [[1, 0, 0, 0, 0, 0, 0, 0]]
labeled_data_sets = []
vocabulary_set = set()

def labeler(sentence, index):
	return sentence, tf.cast(index, tf.int64)

#######################################################
# Label data and update vocabulary
for i, file_name in enumerate(FILE_NAMES):
	lines_dataset = tf.data.TextLineDataset(DIRECTORY + file_name)
	# TODO: iterate lines_dataset only once;
	# 		for now, iterates twice because of decoding bytes issue
	labeled_dataset = lines_dataset.map(lambda sent: labeler(sent, FILE_TAGS[i]))
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

# all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

#######################################################
# Encode vocabulary
# TODO: encode punctuation marks
vocab_size = len(vocabulary_set)
print("Vocabulary size:", vocab_size)
# print(vocabulary_set)
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
	label.set_shape([None])

	return encoded_text, label

all_encoded_data = all_labeled_data.map(encode_map_fn)

#######################################################
# Split dataset into train and test batches
train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))

# Introduce a new vocab of `0`(padding) and `len(vocabulary_set)`(word not in the dictionary)
# TODO: Find effective encoded value for the word that's not in the dictionary, to decrease loss
vocab_size += 1
vocab_size += 1

#######################################################
# Build model
embedding_dim = 128
# TODO: adjust number of LSTM units according to sentence length
# lstm_units = 16
lstm_units = 8


model = tf.keras.Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim))
# TODO: Conv2D + BatchNormalization + ReLU / Pool
model.add(layers.Bidirectional(layers.LSTM(lstm_units)))
model.add(layers.Dense(lstm_units*2, activation='relu'))
# TODO: Dropout
model.add(layers.Dense(len(FILE_NAMES)*8, name='output_layer'))

# TODO: Increase number of labels
model.compile(	optimizer='adam',
				loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
				metrics=['accuracy'])

#######################################################
# Train model
# TODO: adjust epochs as more sentences get added
model.fit(train_data, epochs=2, validation_data=test_data)
eval_loss, eval_acc = model.evaluate(test_data)
print('Evaluation loss: {:.4f}. Evaluation accuracy:{:.4f}'.format(eval_loss, eval_acc))

#######################################################
# Predict sample sentence with the trained model
# sample_sentence = "과일 주문 하 자"
sample_sentence = "과일 주문"
encoded_sentence = encoder.encode(sample_sentence)
print("Encoded sentence:", encoded_sentence)

sample_tensor = tf.constant(np.array(encoded_sentence).reshape(1,len(encoded_sentence)))
print("Sample tensor:", sample_tensor)
print("Predicted label:", model.predict(sample_tensor))
