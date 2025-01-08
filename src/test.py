from flair.models import TextClassifier
from flair.data import Sentence

# Load the pre-trained French sentiment analysis model
model_fr = TextClassifier.load('fr-sentiment') 

# Example French sentence
sentence_fr = Sentence("Ce film était absolument fantastique !") 

# Perform sentiment analysis
model_fr.predict(sentence_fr)

# Get the predicted label 
print(sentence_fr.labels)  # Output: [Label: POSITIVE (score: 0.981)] 