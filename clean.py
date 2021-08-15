#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re, string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import json
import os

class data_clean():
    
    def __init__(self):
        
        print("Data Cleaning Initialized")
        
        
    def remove_urls(self, text):
        """
        Removing htpps links
        """
    
        return re.sub(r"https\://[a-zA-Z0-9\-\.]+.[a-zA-Z]{2,3}(/\S*)?$", " ",text)
    
    
    def remove_html(self, text):
        """
        Remove html tags
        """

        return re.sub(r'<.*?>', ' ', text)
    
    
    def remove_emoji(self, text):
        """
        Remove emojis
        """
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    

    def transcription_sad(self, text):
        """
        Replace some others smileys with SADFACE
        """
        eyes = "[8:=;]"
        nose = "['`\-]"
        smiley = re.compile(r'[8:=;][\'\-]?[(\\/]')
        return smiley.sub(r'SADFACE', text)



    def transcription_smile(self, text):
        """
        Replace some smileys with SMILE
        """
        eyes = "[8:=;]"
        nose = "['`\-]"
        smiley = re.compile(r'[8:=;][\'\-]?[)dDp]')
        return smiley.sub(r'SMILE', text)


    def transcription_heart(self, text):
        """
        Replace <3 with HEART
        """
        heart = re.compile(r'<3')
        return heart.sub(r'HEART', text)

    
    def remove_number(self, text):
        """
        Remove numbers, replace it by NUMBER
        """
        num = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
        return num.sub(r'NUMBER', text)
    
    
    def remove_mention(self, text):
        """
        Remove @ and mention, replace by USER
        """
        at=re.compile(r'@\S+')
        return at.sub(r'USER',text)


    def replace_abbrev(self, text):
        """
        Replace all abbreviations
        """
        
        with open(os.path.join('abbreviations.json'), 'r') as f: # load abbreviation json file
            abbreviations = json.load(f)
        
        def word_abbrev(text):
            """
            Change an abbreviation by its true meaning
            """
            return abbreviations[text.lower()] if text.lower() in abbreviations.keys() else text
        
        string = ""
        for word in text.split():
            string += word_abbrev(word) + " "        
        return string
    

    def remove_elongated_words(self, text):
        """
        Factorize elongated words, add ELONG
        """
        rep = re.compile(r'\b(\S*?)([a-z])\2{2,}\b')
        return rep.sub(r'\1\2 ELONG', text)
    

    def remove_punct(self, text):
        """
        Remove punctuations
        """
        punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" 
        for p in punctuations:
            text = text.replace(p, f' {p} ')

        text = text.replace('...', ' ... ')
        if '...' not in text:
            text = text.replace('..', ' ... ')   
        return text
    
    
    def remove_repeat_punct(self, text):
        """
        Factorize repeated punctuation, add REPEAT
        """
        rep = re.compile(r'([!?.]){2,}')
        return rep.sub(r'\1 REPEAT', text)
    
    
    def remove_all_punct(self, text):
        """
        Remove all punctuations
        """
        table = str.maketrans('','',string.punctuation)
        return text.translate(table)


    def remove_stopwords(self, text):
        
        """
        Removing stopwords from text
        """
        
        stop = set(stopwords.words('english'))
        Punctuation = list(string.punctuation)
        stop.update(Punctuation)
        removed_stop = []

        for i in text.split():
            if i.strip().lower() not in stop:
                removed_stop.append(i.strip())
            
        return " ".join(removed_stop)
    
    def lemmatize_text(self, text):
    
        def get_wordnet_pos(pos_tag):
            if pos_tag.startswith('J'):
                return wordnet.ADJ
            elif pos_tag.startswith('V'):
                return wordnet.VERB
            elif pos_tag.startswith('N'):
                return wordnet.NOUN
            elif pos_tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN
    
        text_l = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tag(text.split(" "))]
    
        return " ".join(text_l)


    def remove_newline(self, text):
        
        """
        Remove all new lines from text
        """
        
        return re.sub("\n", " ", text)


    def remove_space(self, text):
        
        """
        Remove double spaces from text
        """
        
        return re.sub("\s+", " ", text).strip()
    
    
    def white_space(self, text):
        """
        Remove Leading and Tailing white space
        """
        
        return re.sub(r'^\s+|\s+?$', '', text)
    
    
    def contain_oneword(self, text):
        """
        Remove word contain one letter
        """
        
        text_l = [word for word in text.split(" ") if len(word) > 1]
        
        return " ".join(text_l)


    def denoice_text(self, text):
        
        """
        Removing noisy text
        """
        
        text = self.remove_urls(text)
        text = self.remove_html(text)
        text = self.remove_emoji(text)
        text = self.transcription_sad(text)
        text = self.transcription_smile(text)
        text = self.transcription_heart(text)
        text = self.remove_number(text)
        text = self.remove_mention(text)
        text = self.replace_abbrev(text)
        text = self.remove_elongated_words(text)
        text = self.remove_punct(text)
        text = self.remove_repeat_punct(text)
        text = self.remove_all_punct(text)
#         text = self.remove_stopwords(text) # if needed then use remove_stopwords method
        text = self.lemmatize_text(text)
        text = self.remove_newline(text)
        text = self.remove_space(text)
        text = self.white_space(text)
        text = self.contain_oneword(text)
        text = text.lower()
        
        return text