
import streamlit as st
import joblib
import pandas as pd

model=joblib.load('lr_clf.pkl')
vect=joblib.load('vectorizer.pkl')
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

st.title("Movie Review Sentiment Analysis App")
st.subheader(':wave: Hello, people!!')
st.subheader("My name is Manish Khatri, and I recently made a machine learning model to classify the sentiment of a movie review for my Machine Learning project. I thought to create a web application out of it, and here it is.")
st.write("Welcome to the Movie Review Sentiment Analysis App. You have to enter a movie review, and the app will predict its sentiment and display whether you liked the movie or not.")
st.markdown("---")
st.write('''
### Here are some sample reviews, you can copy paste them.
**Pathaan**
I went in to see "pathaan" because of positive RT scores and comments which said this movie reeked of action. The reviewers just left out the part where they were supposed to say the action was unbelievable. It is worth a watch if you have 2.5 hours to kill.
The fight choreography between SRK and 'Jim' (I'm too tired to look up the actor's name) were exceptional. Those featuring Padukone were kind of hit and miss. Sometime they reached Marvel's Black Widow heights, other times they were almost laughable.
Jim was a good villain. His backstory was touching and you could empathize with him and the reasons that he was doing what he was doing. I couldn't root for him because you already know how the story is going to end but you just don't know how they will get there.
Finally, the soundtrack (or background music). It was repetitive when there was a fight between the protagonist and antagonist. So much so that I found it irritating after a while.

**Oppenheimer**
I'm still collecting my thoughts after experiencing this film, Cillian Murphy might as well start clearing a space on his mantle for the Best Actor Oscar. This film is a masterclass in weaving narratives and different time periods while exploring the profound depths of a man whose actions altered the world's trajectory forever, for better or worse. Nolan brings us into the complexities of Oppenheimer, and all the moral conflicts stirring within him. Murphy's portrayal is so riveting that the long run-time became an afterthought. Robert Downey Jr also offers a great performance and Nolan's push and pull with how he uses sound design throughout is the cherry on top. Some viewers might need a brief refresher on WWII and Cold War history, but any film lover should be happy to willingly lose themselves in this film for hours on end.''')
user_input = st.text_area("Enter your movie review:")

if st.button("Predict Sentiment"):
    if user_input:
        review = pd.Series(user_input)
        for key in contractions_dict:
            review = review.str.replace(key, contractions_dict[key])

        review[0] = review[0].lower()
        vect_review = vect.transform(review)

        sentiment = model.predict(vect_review)[0]
        probability = model.predict_proba(vect_review)
        positive_prob = probability[0][1] * 100
        negative_prob = probability[0][0] * 100
        if sentiment == 'POSITIVE':
            st.title(':smile:')
            st.write("<span style='font-size: 24px;'>You liked the movie</span>", unsafe_allow_html=True)
        else:
            st.title(':disappointed:')
            st.write("<span style='font-size: 24px;'>you didn't liked the movie</span>", unsafe_allow_html=True)
        
        st.title("Sentiment Analysis Result")
        st.markdown("---")
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Positive Probability: {positive_prob:.2f}%")
        st.write(f"Negative Probability: {negative_prob:.2f}%")
    else:
        st.warning("Please enter a movie review.")

