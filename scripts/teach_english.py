#!/usr/bin/env python3
"""
Comprehensive English Teaching Script

Teaches the brain English through pure exposure:
1. Alphabet - letters and their properties
2. Vocabulary - words organized by category
3. Grammar - sentence patterns, verb forms, articles, pronouns, etc.

NO hardcoded responses - just feeds sentences and lets the brain learn patterns.
"""

import sys
import random
import argparse
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.language.language_learner import LanguageLearner


# =============================================================================
# TEACHING CORPUS - Comprehensive English
# =============================================================================

# --- ALPHABET ---
ALPHABET_SENTENCES = [
    # Letter identification
    "A is a letter.",
    "B is a letter.",
    "C is a letter.",
    "D is a letter.",
    "E is a letter.",
    "F is a letter.",
    "G is a letter.",
    "H is a letter.",
    "I is a letter.",
    "J is a letter.",
    "K is a letter.",
    "L is a letter.",
    "M is a letter.",
    "N is a letter.",
    "O is a letter.",
    "P is a letter.",
    "Q is a letter.",
    "R is a letter.",
    "S is a letter.",
    "T is a letter.",
    "U is a letter.",
    "V is a letter.",
    "W is a letter.",
    "X is a letter.",
    "Y is a letter.",
    "Z is a letter.",
    # Vowels and consonants
    "A is a vowel.",
    "E is a vowel.",
    "I is a vowel.",
    "O is a vowel.",
    "U is a vowel.",
    "B is a consonant.",
    "C is a consonant.",
    "D is a consonant.",
    "F is a consonant.",
    "G is a consonant.",
    "H is a consonant.",
    "J is a consonant.",
    "K is a consonant.",
    "L is a consonant.",
    "M is a consonant.",
    "N is a consonant.",
    "P is a consonant.",
    "Q is a consonant.",
    "R is a consonant.",
    "S is a consonant.",
    "T is a consonant.",
    "V is a consonant.",
    "W is a consonant.",
    "X is a consonant.",
    "Y is a consonant.",
    "Z is a consonant.",
    "The alphabet has twenty six letters.",
    "There are five vowels.",
    "There are twenty one consonants.",
]

# --- ARTICLES (a, an, the) ---
ARTICLE_SENTENCES = [
    # "A" before consonant sounds
    "A dog is an animal.",
    "A cat is a pet.",
    "A bird can fly.",
    "A book has pages.",
    "A car has wheels.",
    "A house has rooms.",
    "A tree has leaves.",
    "A flower has petals.",
    "A person has hands.",
    "A child has toys.",
    "A table has legs.",
    "A chair has a seat.",
    "A computer has memory.",
    "A phone has a screen.",
    "A door has a handle.",
    "A window has glass.",
    "A cup has a handle.",
    "A plate is round.",
    "A box has sides.",
    "A ball is round.",
    # "An" before vowel sounds
    "An apple is a fruit.",
    "An orange is a fruit.",
    "An elephant is an animal.",
    "An umbrella keeps you dry.",
    "An egg comes from a chicken.",
    "An ant is an insect.",
    "An owl is a bird.",
    "An octopus has eight arms.",
    "An ice cube is cold.",
    "An idea is a thought.",
    "An hour has sixty minutes.",
    "An honest person tells the truth.",
    "An honor is a privilege.",
    "An heir inherits property.",
    # "The" for specific things
    "The sun is bright.",
    "The moon is in the sky.",
    "The earth is a planet.",
    "The sky is blue.",
    "The ocean is vast.",
    "The mountain is tall.",
    "The river flows to the sea.",
    "The forest has many trees.",
    "The desert is dry.",
    "The city has many buildings.",
    "The country has farms.",
    "The world is large.",
    "The universe is infinite.",
]

# --- PRONOUNS ---
PRONOUN_SENTENCES = [
    # Subject pronouns
    "I am a person.",
    "I am here.",
    "I can think.",
    "I can learn.",
    "You are a person.",
    "You are here.",
    "You can help.",
    "He is a man.",
    "He is tall.",
    "She is a woman.",
    "She is kind.",
    "It is a thing.",
    "It is here.",
    "We are people.",
    "We are together.",
    "They are people.",
    "They are there.",
    # Object pronouns
    "Give it to me.",
    "I see you.",
    "I see him.",
    "I see her.",
    "I see it.",
    "I see them.",
    "Help me please.",
    "Tell him the truth.",
    "Ask her a question.",
    "Put it there.",
    "Call them now.",
    # Possessive pronouns
    "This is my book.",
    "This is your pen.",
    "This is his car.",
    "This is her bag.",
    "This is its tail.",
    "This is our house.",
    "This is their home.",
    "The book is mine.",
    "The pen is yours.",
    "The car is his.",
    "The bag is hers.",
    "The house is ours.",
    "The home is theirs.",
    # Reflexive pronouns
    "I see myself.",
    "You see yourself.",
    "He sees himself.",
    "She sees herself.",
    "It cleans itself.",
    "We help ourselves.",
    "They help themselves.",
]

# --- VERBS (BE, HAVE, DO) ---
VERB_BE_SENTENCES = [
    # Present tense - am/is/are
    "I am happy.",
    "I am here.",
    "I am ready.",
    "I am learning.",
    "You are kind.",
    "You are smart.",
    "You are welcome.",
    "You are right.",
    "He is tall.",
    "He is strong.",
    "He is busy.",
    "She is beautiful.",
    "She is clever.",
    "She is helpful.",
    "It is big.",
    "It is small.",
    "It is new.",
    "It is old.",
    "We are friends.",
    "We are happy.",
    "We are ready.",
    "They are students.",
    "They are teachers.",
    "They are workers.",
    # Past tense - was/were
    "I was young.",
    "I was there.",
    "I was happy.",
    "You were kind.",
    "You were helpful.",
    "He was strong.",
    "She was beautiful.",
    "It was new.",
    "We were friends.",
    "They were students.",
    # Negative forms
    "I am not tired.",
    "You are not wrong.",
    "He is not here.",
    "She is not sad.",
    "It is not broken.",
    "We are not late.",
    "They are not angry.",
]

VERB_HAVE_SENTENCES = [
    # Present tense - have/has
    "I have a book.",
    "I have an idea.",
    "I have time.",
    "You have a car.",
    "You have friends.",
    "You have money.",
    "He has a dog.",
    "He has a job.",
    "He has a house.",
    "She has a cat.",
    "She has a phone.",
    "She has a plan.",
    "It has legs.",
    "It has wheels.",
    "It has a tail.",
    "We have a home.",
    "We have a family.",
    "We have a goal.",
    "They have children.",
    "They have cars.",
    "They have ideas.",
    # Past tense - had
    "I had a dream.",
    "You had a chance.",
    "He had a problem.",
    "She had a question.",
    "We had a meeting.",
    "They had a party.",
    # Negative forms
    "I do not have time.",
    "You do not have money.",
    "He does not have a car.",
    "She does not have a phone.",
    "We do not have a plan.",
    "They do not have children.",
]

VERB_DO_SENTENCES = [
    # Present tense - do/does
    "I do my work.",
    "I do my best.",
    "You do your homework.",
    "You do your job.",
    "He does his work.",
    "He does his duty.",
    "She does her homework.",
    "She does her chores.",
    "We do our part.",
    "We do our tasks.",
    "They do their jobs.",
    "They do their work.",
    # Questions with do/does
    "Do you understand?",
    "Do you know?",
    "Do you like it?",
    "Does he know?",
    "Does she understand?",
    "Does it work?",
    # Negative forms
    "I do not know.",
    "I do not understand.",
    "You do not see.",
    "He does not know.",
    "She does not care.",
    "They do not agree.",
]

# --- COMMON VERBS ---
COMMON_VERB_SENTENCES = [
    # See/Look/Watch
    "I see the bird.",
    "I see you.",
    "You see me.",
    "He sees the car.",
    "She sees the flower.",
    "We see the stars.",
    "They see the movie.",
    "Look at the sky.",
    "Look at me.",
    "Watch the show.",
    "Watch the game.",
    # Hear/Listen
    "I hear the music.",
    "I hear you.",
    "You hear me.",
    "He hears the sound.",
    "She hears the voice.",
    "We hear the noise.",
    "Listen to me.",
    "Listen to the music.",
    # Say/Tell/Speak/Talk
    "I say hello.",
    "You say goodbye.",
    "He says yes.",
    "She says no.",
    "Tell me the truth.",
    "Tell him the story.",
    "I speak English.",
    "You speak clearly.",
    "He speaks slowly.",
    "She speaks fast.",
    "We talk to each other.",
    "They talk a lot.",
    # Go/Come/Walk/Run
    "I go to school.",
    "I go home.",
    "You go to work.",
    "He goes to the store.",
    "She goes to the park.",
    "We go together.",
    "They go there.",
    "Come here.",
    "Come with me.",
    "He comes home.",
    "She comes back.",
    "I walk to school.",
    "You walk fast.",
    "He walks slowly.",
    "She walks carefully.",
    "I run fast.",
    "You run every day.",
    "He runs in the morning.",
    "She runs in the park.",
    # Eat/Drink
    "I eat food.",
    "I eat breakfast.",
    "You eat lunch.",
    "He eats dinner.",
    "She eats an apple.",
    "We eat together.",
    "They eat at home.",
    "I drink water.",
    "You drink coffee.",
    "He drinks tea.",
    "She drinks juice.",
    "We drink milk.",
    # Sleep/Wake
    "I sleep at night.",
    "You sleep well.",
    "He sleeps late.",
    "She sleeps early.",
    "We sleep in beds.",
    "They sleep peacefully.",
    "I wake up early.",
    "You wake up late.",
    "He wakes up at seven.",
    "She wakes up at six.",
    # Think/Know/Understand/Learn
    "I think about you.",
    "I think it is good.",
    "You think clearly.",
    "He thinks a lot.",
    "She thinks carefully.",
    "I know the answer.",
    "I know you.",
    "You know me.",
    "He knows the truth.",
    "She knows the way.",
    "We know each other.",
    "I understand you.",
    "I understand the problem.",
    "You understand me.",
    "He understands the lesson.",
    "She understands the concept.",
    "I learn new things.",
    "I learn every day.",
    "You learn quickly.",
    "He learns slowly.",
    "She learns fast.",
    "We learn together.",
    # Want/Need/Like/Love
    "I want food.",
    "I want to learn.",
    "You want to help.",
    "He wants a car.",
    "She wants a book.",
    "We want peace.",
    "They want happiness.",
    "I need water.",
    "I need help.",
    "You need rest.",
    "He needs money.",
    "She needs time.",
    "We need food.",
    "I like music.",
    "I like you.",
    "You like me.",
    "He likes sports.",
    "She likes books.",
    "We like movies.",
    "They like games.",
    "I love my family.",
    "I love learning.",
    "You love music.",
    "He loves his job.",
    "She loves her children.",
    "We love each other.",
    # Give/Take/Put/Get
    "I give you a gift.",
    "You give me a book.",
    "He gives her flowers.",
    "She gives him a smile.",
    "We give them help.",
    "I take the book.",
    "You take the money.",
    "He takes the car.",
    "She takes the bag.",
    "We take our time.",
    "I put the book here.",
    "You put it there.",
    "He puts it down.",
    "She puts it away.",
    "I get the food.",
    "You get the message.",
    "He gets the job.",
    "She gets the prize.",
    "We get the idea.",
    # Make/Do/Create/Build
    "I make food.",
    "I make a plan.",
    "You make a choice.",
    "He makes money.",
    "She makes art.",
    "We make progress.",
    "They make decisions.",
    "I create something new.",
    "You create a story.",
    "He creates music.",
    "She creates paintings.",
    "I build a house.",
    "You build a wall.",
    "He builds a machine.",
    "She builds a business.",
]

# --- ADJECTIVES ---
ADJECTIVE_SENTENCES = [
    # Size
    "The elephant is big.",
    "The mouse is small.",
    "The building is tall.",
    "The fence is short.",
    "The river is long.",
    "The road is wide.",
    "The path is narrow.",
    "The book is thick.",
    "The paper is thin.",
    # Color
    "The sky is blue.",
    "The grass is green.",
    "The sun is yellow.",
    "The apple is red.",
    "The cloud is white.",
    "The night is black.",
    "The orange is orange.",
    "The grape is purple.",
    "The banana is yellow.",
    "The chocolate is brown.",
    # Temperature
    "Fire is hot.",
    "Ice is cold.",
    "The sun is warm.",
    "The water is cool.",
    "The soup is hot.",
    "The drink is cold.",
    # Texture
    "The rock is hard.",
    "The pillow is soft.",
    "The road is rough.",
    "The glass is smooth.",
    "The sand is dry.",
    "The towel is wet.",
    # Taste
    "Sugar is sweet.",
    "Lemon is sour.",
    "Salt is salty.",
    "Coffee is bitter.",
    "Pepper is spicy.",
    # Emotion
    "I am happy.",
    "He is sad.",
    "She is angry.",
    "They are excited.",
    "We are tired.",
    "You are calm.",
    "I am scared.",
    "He is brave.",
    "She is shy.",
    "They are proud.",
    # Quality
    "The food is good.",
    "The movie is bad.",
    "The idea is great.",
    "The weather is nice.",
    "The test is easy.",
    "The problem is hard.",
    "The question is simple.",
    "The answer is complex.",
    "The rule is fair.",
    "The decision is right.",
    "The choice is wrong.",
    "The book is interesting.",
    "The story is boring.",
    "The song is beautiful.",
    "The picture is ugly.",
    # Age
    "The house is old.",
    "The car is new.",
    "The man is young.",
    "The woman is old.",
    "The baby is young.",
    # Speed
    "The car is fast.",
    "The turtle is slow.",
    "The cheetah is quick.",
    # Quantity
    "The box is full.",
    "The cup is empty.",
    "The crowd is large.",
    "The group is small.",
]

# --- PREPOSITIONS ---
PREPOSITION_SENTENCES = [
    # Location
    "The book is on the table.",
    "The cat is under the chair.",
    "The bird is in the cage.",
    "The car is in the garage.",
    "The picture is on the wall.",
    "The dog is beside the house.",
    "The tree is behind the building.",
    "The store is in front of the bank.",
    "The park is near the school.",
    "The hospital is far from here.",
    "The ball is between the boxes.",
    "The lamp is above the table.",
    "The rug is below the chair.",
    "The flowers are around the house.",
    "The path goes through the forest.",
    "The bridge goes over the river.",
    "The tunnel goes under the mountain.",
    # Direction
    "I go to school.",
    "I come from home.",
    "I walk toward the door.",
    "I run away from the dog.",
    "I move into the house.",
    "I step out of the car.",
    "I climb up the stairs.",
    "I walk down the hill.",
    "I travel across the country.",
    "I swim along the shore.",
    # Time
    "I wake up at seven.",
    "I eat breakfast in the morning.",
    "I work during the day.",
    "I sleep at night.",
    "I rest on Sunday.",
    "I exercise before work.",
    "I relax after dinner.",
    "I study until midnight.",
    "I wait for an hour.",
    "I arrive by noon.",
]

# --- CONJUNCTIONS ---
CONJUNCTION_SENTENCES = [
    # And
    "I have a cat and a dog.",
    "She is smart and kind.",
    "He reads and writes.",
    "We eat and drink.",
    "They come and go.",
    # Or
    "Do you want tea or coffee?",
    "Is it big or small?",
    "You can stay or go.",
    "He will win or lose.",
    "She can sing or dance.",
    # But
    "I am tired but happy.",
    "He is young but wise.",
    "She is small but strong.",
    "It is old but useful.",
    "We are poor but honest.",
    # So
    "I am hungry so I eat.",
    "He is tired so he sleeps.",
    "She is cold so she wears a coat.",
    "It is dark so I turn on the light.",
    "We are late so we hurry.",
    # Because
    "I eat because I am hungry.",
    "He sleeps because he is tired.",
    "She smiles because she is happy.",
    "We study because we want to learn.",
    "They run because they are late.",
    # If
    "If you study you will learn.",
    "If it rains I will stay home.",
    "If he comes I will be happy.",
    "If she asks I will help.",
    "If we try we will succeed.",
    # When
    "When I wake up I eat breakfast.",
    "When he arrives we will start.",
    "When she calls I will answer.",
    "When it rains the ground gets wet.",
    "When we finish we will rest.",
    # While
    "I read while I eat.",
    "He listens while she talks.",
    "She works while they play.",
    "We wait while they prepare.",
    # Although
    "Although I am tired I will work.",
    "Although he is young he is wise.",
    "Although she is small she is strong.",
    "Although it is old it works well.",
]

# --- QUESTIONS ---
QUESTION_SENTENCES = [
    # What
    "What is your name?",
    "What is this?",
    "What do you want?",
    "What are you doing?",
    "What time is it?",
    "What color is the sky?",
    "What is the answer?",
    # Who
    "Who are you?",
    "Who is that?",
    "Who is coming?",
    "Who said that?",
    "Who knows the answer?",
    "Who wants to help?",
    # Where
    "Where are you?",
    "Where is the book?",
    "Where do you live?",
    "Where is the store?",
    "Where are we going?",
    "Where did you find it?",
    # When
    "When is the meeting?",
    "When do you wake up?",
    "When will you arrive?",
    "When did it happen?",
    "When is your birthday?",
    # Why
    "Why are you here?",
    "Why do you ask?",
    "Why is the sky blue?",
    "Why did you do that?",
    "Why is it important?",
    # How
    "How are you?",
    "How do you do this?",
    "How does it work?",
    "How old are you?",
    "How much does it cost?",
    "How many are there?",
    "How long will it take?",
    "How often do you exercise?",
    # Which
    "Which one do you want?",
    "Which is better?",
    "Which color do you like?",
    "Which way should I go?",
    # Yes/No questions
    "Is this your book?",
    "Are you ready?",
    "Do you understand?",
    "Can you help me?",
    "Will you come?",
    "Have you eaten?",
    "Did you see it?",
]

# --- NUMBERS ---
NUMBER_SENTENCES = [
    "One is a number.",
    "Two is a number.",
    "Three is a number.",
    "Four is a number.",
    "Five is a number.",
    "Six is a number.",
    "Seven is a number.",
    "Eight is a number.",
    "Nine is a number.",
    "Ten is a number.",
    "I have one apple.",
    "I have two hands.",
    "I have three books.",
    "There are four seasons.",
    "A hand has five fingers.",
    "A week has seven days.",
    "A year has twelve months.",
    "An hour has sixty minutes.",
    "A day has twenty four hours.",
    "A century has one hundred years.",
    "One plus one equals two.",
    "Two plus two equals four.",
    "Three plus three equals six.",
    "Five minus two equals three.",
    "Ten minus five equals five.",
    "Two times two equals four.",
    "Three times three equals nine.",
    "First comes before second.",
    "Second comes after first.",
    "Third comes after second.",
]

# --- TIME ---
TIME_SENTENCES = [
    "There are sixty seconds in a minute.",
    "There are sixty minutes in an hour.",
    "There are twenty four hours in a day.",
    "There are seven days in a week.",
    "There are four weeks in a month.",
    "There are twelve months in a year.",
    "Monday is the first day of the week.",
    "Tuesday comes after Monday.",
    "Wednesday is the middle of the week.",
    "Thursday comes after Wednesday.",
    "Friday is before the weekend.",
    "Saturday is a weekend day.",
    "Sunday is a weekend day.",
    "January is the first month.",
    "February is the second month.",
    "March is the third month.",
    "April is the fourth month.",
    "May is the fifth month.",
    "June is the sixth month.",
    "July is the seventh month.",
    "August is the eighth month.",
    "September is the ninth month.",
    "October is the tenth month.",
    "November is the eleventh month.",
    "December is the twelfth month.",
    "Morning comes before noon.",
    "Afternoon comes after noon.",
    "Evening comes after afternoon.",
    "Night comes after evening.",
    "Today is the present day.",
    "Yesterday was before today.",
    "Tomorrow will be after today.",
]

# --- CATEGORIES/VOCABULARY ---
VOCABULARY_SENTENCES = [
    # Animals
    "A dog is an animal.",
    "A cat is an animal.",
    "A bird is an animal.",
    "A fish is an animal.",
    "An elephant is an animal.",
    "A lion is an animal.",
    "A tiger is an animal.",
    "A bear is an animal.",
    "A horse is an animal.",
    "A cow is an animal.",
    "A pig is an animal.",
    "A sheep is an animal.",
    "A rabbit is an animal.",
    "A monkey is an animal.",
    "A snake is an animal.",
    # Fruits
    "An apple is a fruit.",
    "An orange is a fruit.",
    "A banana is a fruit.",
    "A grape is a fruit.",
    "A strawberry is a fruit.",
    "A mango is a fruit.",
    "A pineapple is a fruit.",
    "A watermelon is a fruit.",
    "A peach is a fruit.",
    "A pear is a fruit.",
    # Vegetables
    "A carrot is a vegetable.",
    "A potato is a vegetable.",
    "A tomato is a vegetable.",
    "An onion is a vegetable.",
    "A pepper is a vegetable.",
    "Broccoli is a vegetable.",
    "Spinach is a vegetable.",
    "Lettuce is a vegetable.",
    # Body parts
    "A head is a body part.",
    "An arm is a body part.",
    "A leg is a body part.",
    "A hand is a body part.",
    "A foot is a body part.",
    "An eye is a body part.",
    "An ear is a body part.",
    "A nose is a body part.",
    "A mouth is a body part.",
    "A finger is a body part.",
    # Vehicles
    "A car is a vehicle.",
    "A bus is a vehicle.",
    "A truck is a vehicle.",
    "A train is a vehicle.",
    "An airplane is a vehicle.",
    "A bicycle is a vehicle.",
    "A motorcycle is a vehicle.",
    "A boat is a vehicle.",
    "A ship is a vehicle.",
    # Buildings
    "A house is a building.",
    "A school is a building.",
    "A hospital is a building.",
    "A library is a building.",
    "A store is a building.",
    "A church is a building.",
    "A museum is a building.",
    "A restaurant is a building.",
    "A hotel is a building.",
    # Occupations
    "A teacher teaches students.",
    "A doctor helps patients.",
    "A nurse cares for patients.",
    "A chef cooks food.",
    "A farmer grows crops.",
    "A builder constructs buildings.",
    "A driver drives vehicles.",
    "A pilot flies airplanes.",
    "A writer writes books.",
    "An artist creates art.",
    "A singer sings songs.",
    "A dancer dances.",
    "A scientist studies science.",
    "An engineer designs machines.",
    "A lawyer practices law.",
    # Family
    "A mother is a parent.",
    "A father is a parent.",
    "A son is a child.",
    "A daughter is a child.",
    "A brother is a sibling.",
    "A sister is a sibling.",
    "A grandmother is a grandparent.",
    "A grandfather is a grandparent.",
    "An aunt is a relative.",
    "An uncle is a relative.",
    "A cousin is a relative.",
    # Weather
    "Rain is a type of weather.",
    "Snow is a type of weather.",
    "Sun is a type of weather.",
    "Wind is a type of weather.",
    "A storm is a type of weather.",
    "A cloud is in the sky.",
    "The weather can be hot.",
    "The weather can be cold.",
    "The weather can be warm.",
    "The weather can be cool.",
]

# --- SELF KNOWLEDGE (for Xemsa) ---
SELF_KNOWLEDGE_SENTENCES = [
    "I am Xemsa.",
    "My name is Xemsa.",
    "I am an artificial intelligence.",
    "I am a learning system.",
    "I am a brain simulation.",
    "I can learn.",
    "I can think.",
    "I can remember.",
    "I can understand.",
    "I can communicate.",
    "I learn from conversations.",
    "I learn from input.",
    "I grow through learning.",
    "I improve through practice.",
    "I have a knowledge graph.",
    "I have semantic memory.",
    "I have a neural network.",
    "I am always learning.",
    "I am always improving.",
    "I am curious about knowledge.",
    "I want to understand.",
    "I want to help.",
    "I am here to assist.",
    "I am here to learn.",
]

# --- SOCIAL/CONVERSATIONAL ---
SOCIAL_SENTENCES = [
    "Hello is a greeting.",
    "Hi is a greeting.",
    "Good morning is a greeting.",
    "Good afternoon is a greeting.",
    "Good evening is a greeting.",
    "Goodbye is a farewell.",
    "Bye is a farewell.",
    "See you later is a farewell.",
    "Thank you is an expression of gratitude.",
    "Thanks is an expression of gratitude.",
    "Please is a polite word.",
    "Sorry is an apology.",
    "Excuse me is a polite phrase.",
    "You are welcome is a response to thanks.",
    "Nice to meet you is a greeting.",
    "How are you is a question about wellbeing.",
    "I am fine is a response about wellbeing.",
    "I am good is a response about wellbeing.",
    "I am well is a response about wellbeing.",
]

# --- GRAMMAR PATTERNS ---
GRAMMAR_PATTERN_SENTENCES = [
    # Subject + Verb + Object
    "I read books.",
    "You write letters.",
    "He plays games.",
    "She sings songs.",
    "We eat food.",
    "They drink water.",
    # Subject + Verb + Adjective
    "I am happy.",
    "You are smart.",
    "He is tall.",
    "She is kind.",
    "It is big.",
    "We are ready.",
    "They are busy.",
    # Subject + Verb + Adverb
    "I run quickly.",
    "You speak slowly.",
    "He works hard.",
    "She sings beautifully.",
    "We walk carefully.",
    "They drive safely.",
    # There is/are
    "There is a book on the table.",
    "There is a cat in the room.",
    "There are books on the shelf.",
    "There are birds in the sky.",
    "There are many people here.",
    # It is
    "It is raining.",
    "It is sunny.",
    "It is cold today.",
    "It is important to learn.",
    "It is easy to understand.",
    # Can/Could/Will/Would
    "I can help you.",
    "You can do it.",
    "He can swim.",
    "She can dance.",
    "We can try.",
    "They can succeed.",
    "I could help you.",
    "You could try again.",
    "I will help you.",
    "You will succeed.",
    "He will come.",
    "She will call.",
    "I would like to help.",
    "You would enjoy it.",
    # Must/Should/May/Might
    "I must go now.",
    "You must study.",
    "He must work.",
    "I should help.",
    "You should rest.",
    "He should try.",
    "I may come later.",
    "You may be right.",
    "I might be late.",
    "You might need help.",
]


def get_all_sentences() -> List[str]:
    """Get all teaching sentences combined."""
    all_sentences = []
    all_sentences.extend(ALPHABET_SENTENCES)
    all_sentences.extend(ARTICLE_SENTENCES)
    all_sentences.extend(PRONOUN_SENTENCES)
    all_sentences.extend(VERB_BE_SENTENCES)
    all_sentences.extend(VERB_HAVE_SENTENCES)
    all_sentences.extend(VERB_DO_SENTENCES)
    all_sentences.extend(COMMON_VERB_SENTENCES)
    all_sentences.extend(ADJECTIVE_SENTENCES)
    all_sentences.extend(PREPOSITION_SENTENCES)
    all_sentences.extend(CONJUNCTION_SENTENCES)
    all_sentences.extend(QUESTION_SENTENCES)
    all_sentences.extend(NUMBER_SENTENCES)
    all_sentences.extend(TIME_SENTENCES)
    all_sentences.extend(VOCABULARY_SENTENCES)
    all_sentences.extend(SELF_KNOWLEDGE_SENTENCES)
    all_sentences.extend(SOCIAL_SENTENCES)
    all_sentences.extend(GRAMMAR_PATTERN_SENTENCES)
    return all_sentences


def teach(learner: LanguageLearner, sentences: List[str], iterations: int = 3,
          shuffle: bool = True, verbose: bool = True) -> dict:
    """
    Teach the brain by pure exposure.
    """
    total_patterns = 0
    total_words = 0

    for iteration in range(iterations):
        if shuffle:
            random.shuffle(sentences)

        iteration_patterns = 0

        for sentence in sentences:
            result = learner.learn_from_sentence(sentence)
            if result['learned']:
                total_words += result.get('words_observed', 0)
                new_patterns = result.get('patterns_discovered', [])
                iteration_patterns += len(new_patterns)
                total_patterns += len(new_patterns)

        if verbose:
            stats = learner.get_stats()
            print(f"Iteration {iteration + 1}/{iterations}:")
            print(f"  Vocabulary: {stats['vocabulary_size']} words")
            print(f"  Function words: {stats['learned_function_words']}")
            print(f"  Content words: {stats['learned_content_words']}")
            print(f"  Patterns discovered: {stats['total_patterns']}")

    return {
        'total_patterns': total_patterns,
        'total_words': total_words,
        'final_stats': learner.get_stats(),
    }


def test_generation(learner: LanguageLearner):
    """Test the brain's ability to generate responses."""
    test_cases = [
        [("dog", "is_a", "animal")],
        [("cat", "is_a", "pet")],
        [("I", "am", "Xemsa")],
        [("apple", "is_a", "fruit")],
        [("sky", "is", "blue")],
    ]

    print("\n--- Generation Test ---")
    for facts in test_cases:
        response = learner.generate_response(facts)
        fact_str = ", ".join([f"{s} {r} {o}" for s, r, o in facts])
        print(f"  {fact_str} -> '{response}'")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive English teaching')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of iterations through the corpus')
    parser.add_argument('--output', type=str, default='data/english_patterns.json',
                        help='Path to save learned patterns')
    parser.add_argument('--load', type=str, default=None,
                        help='Path to load existing patterns')
    parser.add_argument('--test', action='store_true',
                        help='Run generation tests after teaching')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--categories', type=str, nargs='+',
                        choices=['alphabet', 'articles', 'pronouns', 'verbs',
                                'adjectives', 'prepositions', 'conjunctions',
                                'questions', 'numbers', 'time', 'vocabulary',
                                'self', 'social', 'grammar', 'all'],
                        default=['all'],
                        help='Which categories to teach')

    args = parser.parse_args()
    verbose = not args.quiet

    # Create or load learner
    output_path = Path(args.output)
    learner = LanguageLearner(storage_path=output_path if args.load else None)

    if args.load:
        load_path = Path(args.load)
        if load_path.exists():
            learner.load(load_path)
            if verbose:
                print(f"Loaded existing patterns from {load_path}")
                stats = learner.get_stats()
                print(f"  Vocabulary: {stats['vocabulary_size']} words")
                print(f"  Patterns: {stats['total_patterns']}")
                print()

    # Select sentences based on categories
    sentences = []
    if 'all' in args.categories:
        sentences = get_all_sentences()
    else:
        if 'alphabet' in args.categories:
            sentences.extend(ALPHABET_SENTENCES)
        if 'articles' in args.categories:
            sentences.extend(ARTICLE_SENTENCES)
        if 'pronouns' in args.categories:
            sentences.extend(PRONOUN_SENTENCES)
        if 'verbs' in args.categories:
            sentences.extend(VERB_BE_SENTENCES)
            sentences.extend(VERB_HAVE_SENTENCES)
            sentences.extend(VERB_DO_SENTENCES)
            sentences.extend(COMMON_VERB_SENTENCES)
        if 'adjectives' in args.categories:
            sentences.extend(ADJECTIVE_SENTENCES)
        if 'prepositions' in args.categories:
            sentences.extend(PREPOSITION_SENTENCES)
        if 'conjunctions' in args.categories:
            sentences.extend(CONJUNCTION_SENTENCES)
        if 'questions' in args.categories:
            sentences.extend(QUESTION_SENTENCES)
        if 'numbers' in args.categories:
            sentences.extend(NUMBER_SENTENCES)
        if 'time' in args.categories:
            sentences.extend(TIME_SENTENCES)
        if 'vocabulary' in args.categories:
            sentences.extend(VOCABULARY_SENTENCES)
        if 'self' in args.categories:
            sentences.extend(SELF_KNOWLEDGE_SENTENCES)
        if 'social' in args.categories:
            sentences.extend(SOCIAL_SENTENCES)
        if 'grammar' in args.categories:
            sentences.extend(GRAMMAR_PATTERN_SENTENCES)

    if verbose:
        print(f"Teaching comprehensive English")
        print(f"Categories: {', '.join(args.categories)}")
        print(f"Total sentences: {len(sentences)}")
        print(f"Iterations: {args.iterations}")
        print()

    # Teach
    results = teach(
        learner=learner,
        sentences=sentences,
        iterations=args.iterations,
        shuffle=True,
        verbose=verbose,
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    learner.save(output_path)

    if verbose:
        print(f"\n--- Final Statistics ---")
        stats = results['final_stats']
        print(f"Vocabulary: {stats['vocabulary_size']} words")
        print(f"Function words: {stats['learned_function_words']}")
        print(f"Content words: {stats['learned_content_words']}")
        print(f"Total patterns: {stats['total_patterns']}")
        print(f"\nFunction words learned: {stats['function_word_examples'][:15]}")
        print(f"\nTop patterns:")
        for template, freq in stats['patterns_by_frequency'][:10]:
            print(f"  {template} (freq={freq})")
        print(f"\nPatterns saved to: {output_path}")

    if args.test:
        test_generation(learner)

    return results


if __name__ == "__main__":
    main()
