import numpy as np
import torch

class Word2VecGenerator():
    def __init__(self, text_set, word2vec_model, embedding_size=300):
        self.word2vec_model = word2vec_model
        self.embedding_size = embedding_size
        self.train_articles, self.train_questions, self.train_answers, self.train_answer_starts = zip(*text_set)
        #if train_answer_start is -1, then answer is also -1
        self.train_answers = [answer if answer_start != -1 else -1 for answer_start, answer in zip(self.train_answer_starts, self.train_answers)]

    def texts_to_word2vec(self, texts, max_length):
        vectors = []
        for text in texts:
            tokens = text.split()
            text_vectors = [self.word2vec_model[token] for token in tokens if token in self.word2vec_model]
            text_vectors = np.vstack(text_vectors)[:max_length] if len(text_vectors) > 0 else np.zeros((max_length, self.embedding_size))
            vectors.append(np.mean(text_vectors, axis=0))
        return torch.tensor(vectors, dtype=torch.float32)

    def vectorize(self, max_article_length=1200, max_question_length=50):
        article_vectors = self.texts_to_word2vec(self.train_articles, max_article_length)
        question_vectors = self.texts_to_word2vec(self.train_questions, max_question_length)
        return article_vectors, question_vectors, self.train_answer_starts, self.train_answers