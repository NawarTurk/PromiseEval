from flair.data import Sentence
from flair.models import TextClassifier
from transformers import pipeline

# Initialize models
english_model = TextClassifier.load('en-sentiment')
french_model = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment', truncation=True)


def get_sentiment_analysis(text, language='English'):
    if language == 'English':
        sentence = Sentence(text)
        english_model.predict(sentence)
        return sentence.labels[0].value
    elif language == 'French': 
        # truncated_text = truncate_text(text, max_length=512)
        # result = french_model(truncated_text)[0]
        result = french_model(text)[0]

        # Convert 1-5 star rating to POSITIVE/NEGATIVE
        return 'POSITIVE' if float(result['label'].split()[0]) > 3 else 'NEGATIVE'



# from flair.data import Sentence
# from flair.models import TextClassifier

# models = {
#     'English': TextClassifier.load('en-sentiment'),
#     'French': TextClassifier.load('https://huggingface.co/oliverguhr/flair-french-sentiment/resolve/main/final-model.pt')
# }

# def get_sentiment_analysis(text, language='English'):
#     model = models[language]
#     sentence = Sentence(text)
#     model.predict(sentence)
#     return sentence.labels[0].value