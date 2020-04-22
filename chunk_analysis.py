#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

# Chunk analysis example
# Reference: http://konlpy.org/ko/latest/examples/chunking

import konlpy
import nltk

file_in = open("sentences/fruit_order.txt", "r")
# file_out = open("[chunk]fruit_order.txt", "w")
# file_words = open("[words]fruit_order.txt", "w")
file_essential_phrases = open("[essential]fruit_order.txt", "w")

for line in file_in:
	# POS tag a sentence
	# sentence = u'안녕 과일 주문하려는데 오늘 과일 뭐 있어'
	sentence = r'\u' + line.strip()
	words = konlpy.tag._mecab.Mecab().pos(sentence)

	# 	# Remove unicode encodings
	# 	words.pop(0)
	# 	words.pop(0)

	# 	for pair in words:
	# 		word, tag = pair
	# 		file_out.write("{}/{} ".format(word, tag))
	# 		file_words.write("{} ".format(word))
	# 	file_out.write("\n")
	# 	file_words.write("\n")

	# Define a chunk grammar, or chunking rules, then chunk
	# TODO: Modify chunking rules by my needs
	grammar = """
	NP: {<N.*>*<Suffix>?}	# Noun phrase
	VP: {<V.*>*}			# Verb phrase
	"""

	parser = nltk.RegexpParser(grammar)
	chunks = parser.parse(words)
	# print(chunks.pprint())
	
	# Essential [Noun + Verb] phrases
	for subtree in chunks.subtrees():
		phrase_label = subtree.label()
		if phrase_label=='NP' or phrase_label=='VP':
			for e in list(subtree):
				# file_essential_phrases.write("{}/{}".format(e[0], phrase_label))
				file_essential_phrases.write("{} ".format(e[0]))
	file_essential_phrases.write("\n")
	
	# Tag order(pattern) similarity
	'''tags = []
				for leaf in chunks.leaves():
					tags.append(leaf[1])
				# Remove unicode encodings
				tags.pop(0)
				tags.pop(0)
			'''
	# print(tags)

file_in.close()
# file_out.close()
# file_words.close()
file_essential_phrases.close()