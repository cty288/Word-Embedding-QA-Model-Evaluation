from sklearn.feature_extraction.text import TfidfVectorizer
import torch

class TFIDFGenerator():
    def __init__(self, text_set):
        self.train_articles, self.train_questions, self.train_answers, self.train_answer_starts = zip(*text_set)
        #if train_answer_start is -1, then answer is also -1
        self.train_answers = [answer if answer_start != -1 else -1 for answer_start, answer in zip(self.train_answer_starts, self.train_answers)]
        
    
    def texts_to_tfidf(self, texts):
        vectorizer = TfidfVectorizer(max_features=1200)
        vectors = vectorizer.fit_transform(texts).todense().tolist()
        return torch.tensor(vectors, dtype=torch.float32)
            
    def vectorize(self):
        article_vectors = self.texts_to_tfidf(self.train_articles)
        print("article_vectors", article_vectors)
        question_vectors = self.texts_to_tfidf(self.train_questions)
        print("question_vectors", question_vectors)
        return article_vectors, question_vectors, self.train_answer_starts, self.train_answers