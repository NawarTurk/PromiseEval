import flair 

model = flair.models.TextClassifier.load('en-sentiment')

def get_sentiment_analysis(text, model=model):
    sentence = flair.data.Sentence(text)
    model.predict(sentence)
    return sentence.labels[0].value
