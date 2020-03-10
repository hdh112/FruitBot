#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

# Chunk analysis example
# Reference: http://konlpy.org/ko/latest/examples/chunking

import konlpy
# import nltk

file_in = open("fruit_order.txt", "r")
file_out = open("[chunk]fruit_order.txt", "w")

for line in file_in:
	# POS tag a sentence
	# sentence = u'안녕 과일 주문하려는데 오늘 과일 뭐 있어'
	sentence = r'\u' + line.strip()
	words = konlpy.tag._mecab.Mecab().pos(sentence)

	# # Define a chunk grammar, or chunking rules, then chunk
	# # TODO: Modify chunking rules by my needs
	# grammar = """
	# NP: {<N.*>*<Suffix>?}	# Noun phrase
	# VP: {<V.*>*}			# Verb phrase
	# AP: {<A.*>*}			# Adjective phrase
	# """

	# parser = nltk.RegexpParser(grammar)
	# chunks = parser.parse(words)
	# print("# Print whole tree")
	# print(chunks.pprint())

	# Remove unicode encodings
	words.pop(0)
	words.pop(0)

	for pair in words:
		word, tag = pair
		file_out.write("{}/{} ".format(word, tag))
	file_out.write("\n")

file_in.close()
file_out.close()
