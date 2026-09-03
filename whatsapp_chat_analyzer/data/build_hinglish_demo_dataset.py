# build_hinglish_demo_dataset.py
# ---------------------------------
# Generates data/hinglish_sentiment_demo.csv.
#
# *** THIS IS DEMONSTRATION DATA, NOT A VALIDATED RESEARCH DATASET. ***
#
# Why this file exists: the project's original labeled dataset
# (sentiment_train.csv) is English-only and does not exercise the
# Hinglish-aware preprocessing pipeline at all. Since a real WhatsApp
# chat uploaded by a user has NO ground-truth sentiment labels (it is
# personal, unlabeled data -- using it as if it were a labeled training
# set would be scientifically dishonest), we cannot manufacture labels
# from real chats either.
#
# Instead, this script hand-writes a SMALL set of short, code-mixed
# Hinglish/Hindi-in-Roman-script sentences with manually assigned
# positive/negative/neutral labels, purely so the ML pipeline has code
# -mixed examples to train and test against during development. It is:
#   - small (90 rows, 30 per class)
#   - hand-written by the developer, not collected from real users
#   - NOT representative of the full diversity of Hinglish sentiment
#   - NOT to be cited as evidence of real-world model accuracy
#
# Any accuracy/F1 numbers produced using this data describe how well the
# pipeline fits THIS demonstration set only. Swap in a larger, properly
# collected and annotated Hinglish sentiment corpus for any real-world
# claim -- the training code in ml_pipeline.py does not need to change.

import csv
import os

positive = [
    "mast hai bro", "bahut accha laga aaj", "kya baat hai zabardast kaam kiya",
    "yaar tu best hai", "maza aa gaya party mein", "bahut khush hoon aaj",
    "superb yaar ekdum mast", "thik hai sab kuch badiya chal raha hai",
    "bahut badhiya idea hai ye", "dil khush ho gaya ye sunke",
    "kamaal kar diya tune", "sahi hai bhai poora din accha gaya",
    "bindaas mood hai aaj", "full mast party thi kal",
    "bahut proud feel ho raha hai", "ekdum jhakas movie thi",
    "tu rockstar hai yaar", "bahut shukriya madad ke liye",
    "kya scene hai bahut maza aa raha hai", "aaj ka din bahut acha raha",
    "bohot badiya performance thi", "dil se thanks yaar",
    "bahut hi lit party thi", "sabse acchi news hai ye",
    "bahut khushi hui sunke", "wah kya baat hai",
    "tu legend hai bhai", "bahut sukoon mila aaj",
    "superb result aaya hai", "bahut mast weather hai aaj",
]

negative = [
    "mood off hai yaar", "aaj bahut bakwas din tha", "bahut tension ho rahi hai",
    "yeh bilkul bekar tha", "bahut gussa aa raha hai mujhe", "sab kuch kharab ho gaya",
    "bahut udaas hoon aaj", "yeh scene bilkul faltu tha", "bahut thak gaya hoon aaj",
    "kal ka din bahut bura tha", "bahut pareshan hoon is baat se", "dil bahut dukhi hai",
    "bahut niraash hoon result se", "yeh bilkul galat hua", "aaj sab kuch bigad gaya",
    "bahut stress ho raha hai exam ka", "bura laga ye sunke", "bahut disappointing tha match",
    "mood kharab hai bilkul", "kal se sab kuch off chal raha hai",
    "bahut irritating tha wo insaan", "bahut rona aa raha hai",
    "sab bekar lag raha hai aajkal", "bahut takleef ho rahi hai",
    "yeh bahut hi ghatiya tha", "bahut akela mehsoos ho raha hai",
    "sab plan fail ho gaya", "bahut naraz hoon tumse",
    "bahut bura laga sun ke", "dil toot gaya ye sunke",
]

neutral = [
    "kal milte hai", "kitne baje aana hai", "class kal se shuru hogi",
    "meeting reschedule ho gayi hai", "file bhej do please", "kal exam hai subah nau baje",
    "lunch kab karoge", "tum kaha ho abhi", "weekend pe kya plan hai",
    "group mein sab log aa jao", "assignment submit karna hai kal tak", "train subah dus baje hai",
    "address bhejo please", "call kar lena jab free ho", "schedule confirm kar do",
    "list bhej raha hoon dekh lena", "location share karo please", "next hafte milte hai",
    "project ka status kya hai", "form fill karke bhejna hai", "kal ka agenda kya hai",
    "store kitne baje band hota hai", "payment kab tak karni hai", "venue change ho gaya hai",
    "update bhej dena jab ho jaye", "budget discuss karna hai kal", "presentation ready hai kya",
    "invoice bhejna mat bhoolna", "class online move ho gayi hai", "schedule ek baar phir se check karo",
]

rows = (
    [(t, "positive") for t in positive]
    + [(t, "negative") for t in negative]
    + [(t, "neutral") for t in neutral]
)

out_path = os.path.join(os.path.dirname(__file__), "hinglish_sentiment_demo.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    f.write(
        "# DEMONSTRATION DATA ONLY -- hand-written by the developer for pipeline "
        "development/testing. Small (90 rows), not collected from real users, and "
        "NOT representative of general Hinglish sentiment. See "
        "build_hinglish_demo_dataset.py header for full context.\n"
    )
    w = csv.writer(f)
    w.writerow(["text", "label"])
    w.writerows(rows)

print(f"Wrote {len(rows)} demonstration-labeled Hinglish rows -> {out_path}")
