from sklearn.feature_extraction.text import TfidfVectorizer
import torch

class TFIDFGenerator():
    def __init__(self, text_set):
        self.train_articles, self.train_questions, self.train_answers, self.train_answer_starts = zip(*text_set)
        
    
    def texts_to_tfidf(self, texts):
        vectorizer = TfidfVectorizer(max_features=300)
        vectors = vectorizer.fit_transform(texts).todense().tolist()
        return torch.tensor(vectors, dtype=torch.float32)
            
    def vectorize(self):
        article_vectors = self.texts_to_tfidf(self.train_articles)
        question_vectors = self.texts_to_tfidf(self.train_questions)
        return article_vectors, question_vectors, self.train_answer_starts