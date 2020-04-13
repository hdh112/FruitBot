# FruitBot
Empathy chatbot implementation that orders fruits.

## Target user
Students & one-person households, to maintain physical/mental health
* When living alone, one can easily become less aware of taking care of health, or easily feel lonely.
* With _FruitBot_, you're not alone. Because it is there beside you, you become aware that you have to eat good and be healthy :)

## Features
### Informal tone
* _FruitBot_ uses informal tone of Korean language - the tone used between friends.
    * This setting was intended to consider the chatbot as a **friend**, rather than a passive assistant.
* Examples:
```
안녕!        //  Hi!
어서와 :D    //  Welcome :D
반가워~      //  Good to see you :)
```
```
지금 주문할 수 있는 과일은 다음과 같아!
[방울토마토 / 복숭아 / 블루베리 / 사과 / 자몽 / 체리 / 컵과일 / 포도 / 딸기]

For now, you can order the following fruit options!
[Cherry tomatoes / Peaches / Blueberries / Apples / Grapefruits / Cherries / Fruit cups / Grapes / Strawberries]
```
```
고민 된다면 기분이나 필요에 따라 아래 특징 중에 얘기해주면, 그에 맞춰 추천해줄게 ㅎㅎ
[간편함 / 가격 / 양 / 크기 / 맛]

If nothing particular comes up in your mind, you can tell me what you want or need according to the following. I'll recommend an option based on it :D
[Convenience / Price / Quantity / Size / Taste]
```

### Does not force to order
* _FruitBot_ does not always lead the user to order fruits.
    * This way, the user feels **less pressure** when talking with _FruitBot_.
        * The user can just chat with _FruitBot_, and does not always need to order fruits when having a conversation.
    * As users have less burden, this leads users to **chat more frequently** with _FruitBot_.
    
### Empathize to users having a hard time
* _FruitBot_ empathizes to the user's status, especially when the user is having a hard time.
* Examples of user status:
```
sad: 슬퍼 ㅠㅠ / 기운이 없어 / 무기력해 / 바빠 / 시간이 없어 ㅠㅠ / 힘들어 / 우울해
tired: 피곤해 / 쉬고 싶어
sick: 아파 / 몸이 허해
```

### Korean language
* Currently, _FruitBot_ is implemented in Korean, since my(Doheon Hwang's) mother tongue is in Korean. I could most naturally express the above intentions in Korean.
* Should any of you want to contribute in another natural language, you are more than welcome!

## Scope
So far, what _FruitBot_ plans to say is described [here](https://github.com/hdh112/FruitBot/issues/3#issuecomment-601589727).\
Once the implementation is complete, this scope may be extended further.

## Tools
* Language: Python
* Machine learning library: [Tensorflow](https://www.tensorflow.org/)
* Korean preprocessing package: `Mecab()` of [KoNLPy](http://konlpy.org/en/latest/)

## Directory specification
* ./[chunk_analysis.py](https://github.com/hdh112/FruitBot/blob/master/chunk_analysis.py): Splits sentences into chunks, according to tags(ex: noun, verb, etc.). The splitter utilizes `konlpy.Mecab()`.
* ./[classifying_model.py](https://github.com/hdh112/FruitBot/blob/master/classifying_model.py): Classifies sentences according to intent. The classifier utilizes Tensorflow.
* ./sentences/*: Contains input sentences.
* ./words/*: Contains sentences splitted into words.
* ./chunks/*: Contains sentences splitted into chunks, whereas a chunk is a tagged word.
* ./analysis/*: Contains analysis notes on input sentences.

## Contributors
* Doheon Hwang - [hdh112](https://github.com/hdh112/)
