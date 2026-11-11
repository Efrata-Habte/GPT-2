# Experiment 2: prompt length and context

Checkpoint `app/checkpoints/english/model.pt`, 500 held-out windows of 256 tokens.

## A. Loss by position in the window
![loss by position](exp2_loss_by_position.png)

| positions | mean loss |
|---|---|
| 0-0 | 2.5212 |
| 1-7 | 1.6298 |
| 8-31 | 1.4661 |
| 32-63 | 1.4625 |
| 64-127 | 1.4474 |
| 128-255 | 1.4570 |

## B. Loss on the 32 tokens after an L-token prompt

| prompt length L | mean loss on next 32 tokens |
|---|---|
| 1 | 1.5349 |
| 2 | 1.4997 |
| 4 | 1.4653 |
| 8 | 1.4572 |
| 16 | 1.4702 |
| 32 | 1.4669 |
| 64 | 1.4448 |
| 128 | 1.4532 |

## C. Generations from the same passage with different prompt lengths

### prompt length 8 tokens
Prompt:
```
like a s
```
*greedy:*
```
trange of the state,
And then the state of the sea of the sea
Where they shall be their state. The senate of them
The senate of the season of their state,
Whose they shall be their strength that they 
```
*T=0.8:*
```
lave, her deed,
Either clouds and in one of eye,--

ROMEO:
Give me that a minster servant have;
That my mouth hath company 'God of the queen!'
And methinks I mine ears of the prayers
May be grave's be
```

### prompt length 32 tokens
Prompt:
```
 sir, methinks you walk like a s
```
*greedy:*
```
trange of the world,
And then the sea of the sea of the sea
Whose senate was the sea of the season of the world,
Whose sense state and the season of the sea
Whose several strokes and the season of the
```
*T=0.8:*
```
lave, here dear
Either clouds was in one of eye,--

Second Gentleman:
That's sentenced, your honour, my mother, sir.

First Gentleman:
There is the every noble thief and good to the people:
I see, I w
```

### prompt length 128 tokens
Prompt:
```
ept his service.

BAPTISTA:
A thousand thanks, Signior Gremio.
Welcome, good Cambio.
But, gentle sir, methinks you walk like a s
```
*greedy:*
```
trange to the word.

CAMILLO:
I would they say 'tis true.

CAMILLO:
I would they say 'tis the matter.

CAMILLO:
I would they say 'tis the matter.

CLAUDIO:
The matter that I have seen them so far that
```
*T=0.8:*
```
lave,
Set deep a fair root as each another gentleman:
I do come to sit a main on my country:
Speak not thou servicess laid, as thou art not like too:
The nobility storms of truth. What still thou, I w
```
