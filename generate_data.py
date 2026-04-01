import pandas as pd
import random

airlines = ['Virgin America', 'United', 'Southwest', 'Delta', 'US Airways', 'American']
sentiments = ['positive', 'negative', 'neutral']

positive_phrases = [
    "I had a wonderful flight experience today! The crew was so polite.",
    "Thank you for the amazing service.",
    "Best flight ever. Will fly with you again.",
    "Super fast and efficient boarding process. Loved it!",
    "The food was surprisingly good.",
    "Great customer service at the gate.",
    "Smooth landing and great pilot.",
    "Love the in-flight entertainment.",
    "Very comfortable seats and plenty of legroom."
]

negative_phrases = [
    "Worst experience of my life. Flight delayed by 4 hours.",
    "Lost my luggage again... extremely disappointed.",
    "The staff was incredibly rude and unhelpful.",
    "Terrible customer service on the phone.",
    "Why is my flight always delayed without any notice?",
    "Seats are too small and uncomfortable.",
    "Food was awful, could not eat anything.",
    "Horrible boarding process, chaotic and stressful.",
    "Never flying with you again. Absolutely terrible."
]

neutral_phrases = [
    "Flight was okay, nothing special.",
    "Arrived on time. Standard experience.",
    "Average service. Could be better.",
    "Just another flight.",
    "Boarding was standard.",
    "Food was okay, pretty standard for airline food.",
    "Got from A to B safely."
]

data = []
for _ in range(1500):
    airline = random.choice(airlines)
    sentiment = random.choice(sentiments)
    
    if sentiment == 'positive':
        text = random.choice(positive_phrases) + " @{} #amazing".format(airline.replace(" ", ""))
    elif sentiment == 'negative':
        text = random.choice(negative_phrases) + " @{} #awful".format(airline.replace(" ", ""))
    else:
        text = random.choice(neutral_phrases) + " @{}".format(airline.replace(" ", ""))
        
    data.append([airline, sentiment, text])

df = pd.DataFrame(data, columns=['airline', 'airline_sentiment', 'text'])
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("Tweets.csv", index=False)
print("Generated Tweets.csv successfully.")
