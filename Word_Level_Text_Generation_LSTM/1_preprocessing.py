import numpy as np
import string
from string import maketrans

# Read document
filename = 'republic.txt'
file = open(filename, 'r')
document = file.read()
file.close()

# Clean text
document = document.replace('--', ' ')
document = ' '.join(document.split())
exclude = set(string.punctuation)
document = ''.join(ch for ch in document if ch not in exclude)
tokens = document.split(' ')
tokens = [word for word in tokens if word.isalpha()]
tokens = [word.lower() for word in tokens]
print tokens[:20]
print len(tokens)

# Prepare data each sequence of 51 words.
length = 51
sentences = []
for i in range(length, len(tokens)):
	l = ' '.join(tokens[i-length:i])
	sentences.append(l)
print len(sentences)

# Save the file
filename = 'rebuplic_preprocessed.txt'
file = open(filename, 'w')
file.write('\n'.join(sentences))
file.close()
