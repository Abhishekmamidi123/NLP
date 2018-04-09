from keras.preprocessing.text import Tokenizer

docs = ['TajMahal is in India', 'Human beings cannot live is without oxygen', 'Robots are very useful now a days', 'Water is used for drinking']

t = Tokenizer()
t.fit_on_texts(docs)

# Summary
print t.word_counts # Count of words
print t.document_count # Number of counts
print t.word_index # Index of words
print t.word_docs # How many times the word occurred in the documents

encoded_docs = t.texts_to_matrix(docs, mode='count')
print encoded_docs

encoded_docs_numbers = t.texts_to_sequences(docs)
