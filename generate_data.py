import pandas as pd
import random

airlines = ['Virgin America', 'United', 'Southwest', 'Delta', 'US Airways', 'American']
sentiments = ['positive', 'negative', 'neutral']

positive_phrases = [
    "I had a wonderful flight experience today! The crew was so polite.",
    "Thank you for the amazing service.", "Best flight ever. Will fly with you again.",
    "Super fast and efficient boarding process. Loved it!", "The food was surprisingly good.",
    "Great customer service at the gate.", "Smooth landing and great pilot.",
    "Love the in-flight entertainment.", "Very comfortable seats and plenty of legroom.",
    "Amazing flight, very smooth!", "Great service from the cabin crew. #happy",
    "Enjoyed the meal on board.", "Quick and easy check-in.",
    "Really appreciate the help at the gate!", "Best airline I've flown with so far.",
    "The in-flight entertainment was top notch.", "Thank you for the early arrival!",
    "Friendly staff and clean plane.", "Excellent experience from start to finish.",
    "Professional and courteous crew members.", "Easy booking process and great prices.",
    "Modern fleet and very quiet cabin.", "Top-tier reliability and on-time performance.",
    "The lounge was luxurious and relaxing.", "Highly recommend this airline for long-haul.",
    "Exuberant and helpful flight attendants throughout.", "Seamless connection in the hub airport.",
    "A premium feel from boarding to deplaning.", "Incredible value for the service provided.",
    "The mobile app experience was very intuitive."
]

negative_phrases = [
    "Worst experience of my life. Flight delayed by 4 hours.",
    "Lost my luggage again... extremely disappointed.", "The staff was incredibly rude and unhelpful.",
    "Terrible customer service on the phone.", "Why is my flight always delayed without any notice?",
    "Seats are too small and uncomfortable.", "Food was awful, could not eat anything.",
    "Horrible boarding process, chaotic and stressful.", "Never flying with you again. Absolutely terrible.",
    "Flight delayed by 6 hours. Unacceptable!", "Lost my bags again. Horrible service.",
    "The staff was extremely rude to me.", "Terrible experience, would NOT recommend.",
    "My flight was cancelled and no help was offered.", "Very cramped seats, no legroom at all.",
    "The food was disgusting and cold.", "Waiting for ages at the baggage claim.",
    "Customer support is useless and unhelpful.", "Broken seat and no working entertainment.",
    "Still waiting for a refund for my cancelled flight.", "The plane was dirty and smelled bad.",
    "The cabin was super late and super filthy.", "Luggage arrival was delayed significantly.",
    "The check-in process was incredibly slow and taxing.", "The crew seemed completely unmotivated today.",
    "An appalling experience from start to finish.", "The tray tables were filthy and the seats broken.",
    "Total disaster, I missed my connecting flight.", "Incompetent staff at the gate and on board.",
    "Stay away from this airline at all costs."
]

neutral_phrases = [
    "Flight was okay, nothing special.", "Arrived on time. Standard experience.",
    "Average service. Could be better.", "Just another flight.", "Boarding was standard.",
    "Food was okay, pretty standard for airline food.", "Got from A to B safely.",
    "Flying to NYC today.", "Checked in my bags at the counter.", "Is flight AB123 on time?",
    "What is the baggage allowance for economy?", "Just boarded the plane.",
    "Landed safely in London.", "Standard flight, nothing special.",
    "Can I change my seat assignment?", "Where is the lounge located?",
    "Inquiry about my flight status.", "Looking for my flight number.",
    "Is the check-in counter open yet?", "Can you please DM me regarding my reservation?",
    "Checking the status of my connection.", "I need to upgrade my seat.",
    "How do I access the Wi-Fi on board?", "My flight number is DL456.",
    "Are there any delays for the weather?", "Standard economy class experience.",
    "Just landed at LAX.", "Waiting for my boarding group.", "Connecting in Chicago.",
    "Where is the terminal 2 gate?"
]

data = []
for _ in range(5000):
    airline = random.choice(airlines)
    sentiment = random.choice(sentiments)
    
    # Generate 1-4 sentences per review for complexity
    num_sentences = random.randint(1, 4)
    if sentiment == 'positive':
        phrases = random.sample(positive_phrases, min(num_sentences, len(positive_phrases)))
        text = " ".join(phrases) + " @{} #amazing".format(airline.replace(" ", ""))
    elif sentiment == 'negative':
        phrases = random.sample(negative_phrases, min(num_sentences, len(negative_phrases)))
        text = " ".join(phrases) + " @{} #awful".format(airline.replace(" ", ""))
    else:
        phrases = random.sample(neutral_phrases, min(num_sentences, len(neutral_phrases)))
        text = " ".join(phrases) + " @{}".format(airline.replace(" ", ""))
        
    data.append([airline, sentiment, text])

df = pd.DataFrame(data, columns=['airline', 'airline_sentiment', 'text'])
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("Tweets.csv", index=False)
print("Generated 5000 complex Tweets.csv successfully.")
