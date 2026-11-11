# Experiment 12: small vs full training data

Same architecture, same number of steps and the same seed; only the training data differs. Both models of a pair are scored on the same held-out text that neither saw during training (`data/test_sets/`).

## English

| model | training chars | params | test bits/char: unseen Shakespeare | valid words (T=0.8) | distinct-2 |
|---|---|---|---|---|---|
| small data: Tiny Shakespeare | 1,003,854 | 10,770,816 | 2.311 | 97.7% | 0.99 |
| full data: Complete Works | 4,347,162 | 10,770,432 | 1.778 | 98.0% | 0.99 |

Full-data model vs small-data model on the shared test text: unseen Shakespeare: 23.1% better. Characters removed from the test text because one model never saw them: unseen Shakespeare: 0.00%.

**small data: Tiny Shakespeare** (prompt `ROMEO:`, T=0.8, seed 1337)
```
ROMEO:
Clarence before than the chastitors that art words.
Is it tell it? thou comest on the joy of thee?

HASTINGS:
Have we been upon thy shame against the house,
Unless those gadly not thy success?

GLOUCESTER:
I have such a word to bear with their spost sweet.

GLOUCESTER:
They from my deserves he hath prefer'd thee from his life.

LADY ANNE:
Peace, so may I love a churchyard.
Why now they must stand 
```

**full data: Complete Works** (prompt `ROMEO:`, T=0.8, seed 1337)
```
ROMEO:
Ay, thou art to say I to offence thou say'st,
And thou tend'st my fortune from business.

ROMEO:
There is my lady that favours doth put it me.

PANDARUS:
Ay, my good lord, thy lord, well sent her
To th' expectation of my heart.

ROMEO:
Not so, I bid it ride it.

ROMEO:
Prove I thee say she is mine, thou wilt not find
The best knight in the country of my purpose,
Let those great conquering charges 
```

## Amharic

| model | training chars | params | test bits/char: news (held out) | test bits/char: Wikipedia (held out) | valid words (T=0.8) | distinct-2 |
|---|---|---|---|---|---|---|
| small data: news | 3,375,548 | 461,440 | 3.072 | 4.188 | 88.2% | 1.00 |
| full data: Wikipedia + news | 9,909,215 | 476,160 | 3.252 | 3.525 | 83.3% | 1.00 |

Full-data model vs small-data model on the shared test text: news (held out): 5.9% worse, Wikipedia (held out): 15.9% better. Characters removed from the test text because one model never saw them: news (held out): 0.00%, Wikipedia (held out): 0.35%.

**small data: news** (prompt `የኢትዮጵያ`, T=0.8, seed 1337)
```
የኢትዮጵያ ሚንስትሩ ገቢ አሲተያየቱም የተባሉት እንደሚሆን ከፍተኛ ነው። የቀድሞው ስምምነት ቡድኑ መመተው የነሐሴ ጨዋታ ለህዝብ የመጨረሻ አጣጣሪ ማዕከላት በአዋቂ ደግሞ ባለፈው ዓመት ላይ ሲሆን በማጠናቀቅ አዳዲስ አደጋ ወደዚህ ውይይት ብቻ ተከትሎ መካከል ነው። አፍሪካ ፕሬዝዳንት ይህ የሆነው በምርጫ ተመልክቷል። መሞታቸውን ጠቅላይ ሚኒስትር የሚያስፈልገው ዶ/ር እና ዲሞክራሲያ ሁኔታ አትሌቶች በበኩላቸውን እንዲሁም መቃወም በቤተሰቦቻቸው አሊያም ተደርጓል። ከሰላም ጋር በአሜሪካ አገሪቱን የሚያስወጣው ክትባት እንዲዘገባ ከአስርት የዋጋ ሁኔታ ያደረጉ ወደ 15 ብር ለኢትዮጵያ ለቢቢሲ ከፍተኛ ያለው የሚወደው ግራች የሆኑት የካናዳ የተባለው በፌ
```

**full data: Wikipedia + news** (prompt `የኢትዮጵያ`, T=0.8, seed 1337)
```
የኢትዮጵያ ዓመት ሕዝብ እና እርግጥ እርግጥ እንዳገሩ የሰከተበ በላይ የሚያገኝባቸው ሳይሆን አገሪቱ ትልቅ ሴቶች በመሆኑን ሲል እሳት በመደበቅት መሄድ ዱኔራ ወደ አንድን ይባላል። ስለዚህ ጨዋታው ለገለጸው የሆነው አባላት ምክንያት ነው። አንዳንድ የሚቀበለ ከመምህርት የተመለከተ የተሻለ መንገድ የሚል ገለፀው ምክንያት ክርስቶስ እንደነበረው እንደመታት በሌሎች አንዱ ወይም የአባል በላይ በመሬት ዩኒቨርሲቲ ያደርገው የሚባል መንግሥት በትሮጵያ በመቶ የመንገዶች ላይ ከቴክኖሎጂ እየተለያዩ በተጠቁም መለየትምህርት በ190 እንደሚገኙበት የጉዳይነት መስመርኛ ተግባራት እና የኩባም ውስጥ ወደ 1434 ነው። በእያንዳንዱ ይህ ነው። "በ2000 የምድብ በ2
```

![data scale](data_scale.png)

Caveat: one seed per model; with a fixed number of steps the full-data model also sees each character fewer times (less memorisation), which is part of the effect being measured.
