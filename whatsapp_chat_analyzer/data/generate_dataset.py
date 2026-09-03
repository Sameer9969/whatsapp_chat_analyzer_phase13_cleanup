# Generates data/sentiment_train.csv - a hand-curated labeled dataset of
# short chat-style messages (English + common Hinglish chat phrasing),
# used to train the sentiment classifier offline (no external download needed).
import csv

positive = [
    "this is amazing news", "congrats on the win", "so happy for you today",
    "great job everyone", "thanks so much for your help", "that's wonderful to hear",
    "I love this idea", "feeling super relieved and happy", "what a fantastic day",
    "you guys are the best", "proud of us honestly", "this made my day",
    "awesome work team", "I'm really excited about this", "such good news",
    "that's hilarious lol", "can't stop laughing at this", "best lunch ever",
    "thank you for the reminder", "great teamwork everyone", "yay we did it",
    "this is so much fun", "I appreciate you a lot", "well done to everyone",
    "feeling grateful today", "that's a great read thanks for sharing",
    "count me in too excited", "on it thanks", "all good just finished my exam",
    "nice one bro party time", "amazing performance today", "super excited for this",
    "this is exactly what I needed", "great to see you", "love spending time with you all",
    "fantastic idea let's do it", "so proud of this team", "wonderful surprise",
    "everything worked out perfectly", "what great news to wake up to",
    "this project turned out really well", "happy to help anytime",
    "such a relief it's finally done", "great vibes today", "loving this weather",
    "you did an incredible job", "this deserves a celebration", "so grateful for this",
    "such a positive update", "excited to start this journey", "feeling blessed today",
]

negative = [
    "I'm really disappointed with today", "this is so frustrating", "feeling stressed about deadlines",
    "that was such a boring lecture", "this traffic is horrible", "I hate waiting like this",
    "so sad about the results", "this is really annoying", "worst day ever",
    "I completely forgot about it", "feeling exhausted and upset", "this is a disaster",
    "nothing is going right today", "I'm so tired of this", "this makes me angry",
    "such a terrible experience", "I regret doing that", "everything is falling apart",
    "this is unacceptable", "I feel so let down", "that was a huge mistake",
    "I can't deal with this anymore", "so much stress this week", "this is really upsetting",
    "I'm devastated by this news", "this project is a mess", "feeling hopeless right now",
    "that response was so rude", "I'm furious about this", "this is such a letdown",
    "worst lecture I've ever attended", "I'm sick of these delays", "this update is disappointing",
    "everything feels overwhelming", "I'm annoyed with the plan", "this is really sad news",
    "such an awful outcome", "I feel ignored and hurt", "this keeps getting worse",
    "I'm exhausted from all this", "that was completely unfair", "this is so irritating",
    "I dislike how this turned out", "feeling anxious about tomorrow", "this is heartbreaking",
    "such a rough week", "I'm frustrated with the delay", "this really upset me",
    "that was a horrible mistake", "I'm dreading this deadline",
]

neutral = [
    "what time is the meeting", "let me know when you're free", "sending the file now",
    "reminder submission is due tonight", "check this out", "anyone free for lunch today",
    "good morning everyone", "here is the link", "please review this document",
    "the event starts at 5pm", "I'll be there in ten minutes", "can you share the notes",
    "meeting has been rescheduled", "here's the report for this month", "let's discuss this tomorrow",
    "the file has been uploaded", "please confirm your attendance", "this is the agenda for today",
    "we need to finalize the plan", "the deadline is next Friday", "sharing the update here",
    "let's plan for next week", "please find attached the document", "the class starts at nine",
    "will update you once done", "here is today's schedule", "the store closes at eight",
    "let's connect over a call", "please check your email", "the assignment covers chapter five",
    "I'll send the details shortly", "the venue has changed", "just checking in on this",
    "note this down for later", "the form needs to be submitted", "please share your availability",
    "the results will be out tomorrow", "we should review the budget", "here's a quick summary",
    "the train arrives at noon", "please update the spreadsheet", "let's confirm the schedule",
    "the payment is due this week", "kindly share the location", "the presentation is ready",
    "we can discuss this later", "please send the invoice", "the report is attached below",
    "let's sync up tomorrow morning", "the class has been moved online", "here's what we discussed",
]

rows = [(t, "positive") for t in positive] + [(t, "negative") for t in negative] + [(t, "neutral") for t in neutral]

with open("sentiment_train.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["text", "label"])
    w.writerows(rows)

print(f"Wrote {len(rows)} labeled rows")
