# Experiment 1: temperature

Checkpoint `checkpoints/demo_english/ckpt_best.pt`, prompt `'ROMEO:'`, 8 samples x 400 new tokens per setting, seed 1337 (same seed for every setting). Metrics are averaged over samples. Greedy gives one deterministic text, so all its samples are identical.

| setting | distinct-1 | distinct-2 | repetition-4 | valid words |
|---|---|---|---|---|
| greedy | 0.105 | 0.165 | 0.675 | 1.000 |
| T=0.2 | 0.358 | 0.663 | 0.046 | 0.972 |
| T=0.5 | 0.597 | 0.956 | 0.000 | 0.970 |
| T=0.8 | 0.770 | 0.995 | 0.000 | 0.885 |
| T=1.0 | 0.850 | 0.998 | 0.000 | 0.828 |
| T=1.5 | 0.930 | 1.000 | 0.000 | 0.576 |

![temperature](exp1_temperature.png)

## Sample 1 for each setting (first 400 characters)

**greedy**
```
ROMEO:
I will the stand the stand of the stand of the stand the stands
That the stand the stand of the stand of the stands,
And the stand of the stand of the death,
And the stand that the stand of the stand of the death,
And the stand that the stand of the stand of the death,
And the stand the stand of the stand of the death,
And the stand that the stand of the stand of the death,
And the stand that the
```

**T=0.2**
```
ROMEO:
I will be so should the present of the death,
And the contraction of the lady of the death,
And the tread of his father state of the world of the sealth,
And the will the world hath of the world of the prince,
And the stand that the provereign of the bearts of the duke
That of the proceed of the be some of the part,
And the set the confence of the courtent of the father,
That the shall be stand t
```

**T=0.5**
```
ROMEO:
The present to on the world that from and life of his cours.

LUCIO:
The life with had not so show stand the will and from to leave
That his the hand was to the neither?

LUCIO:
I am they know should me the not to more, and the world
And be venger from my son of the consuld of more unto to the king live,
To have the seasure set all the common of will in my life.

FLORIZEL:
I am the king, the lost
```

**T=0.8**
```
ROMEO:
The present me of your sorrowless of man,
Who so? Camily your good marrel hath the had news'd.

ISABELLA:
It will a truth.

Third Citizen:
What, he speak unto the nature.

DUKE VINCENTIO:
The souls rememberant interning away;
I would to underledge for my lord: thou requittle this cruly
Thou not weep the wailst of like as I find; like the smill'd will is?

SICINIUS:
You are to here to us, the ligh
```

**T=1.0**
```
ROMEO:
The pity is much sations of that from and justly?
Come, let knock and is like with his neck's.

ISABELLA:
It will a truth.

TRANIO:
Citizens toes me peace their non,
That with is hath would me suopsrenself an piltient.
Our hath wold fooluth veens may:
My lord of there,
Intime his cruly thou not weep so.
This signiorates. What, silrown? What he how I
will this to of body gold.

Nurse:
Bing let my 
```

**T=1.5**
```
ROMEO:
TElBz-day, then my fortline ell: freen!

LEONTES:
CloWe'll lock is armies; I prince gentleman.

ISABELLA:
It with a truth.

TRANIO:
Citizes
My ember's madutiege, ni, great bebail:--Do wought I slow
reise? Manted then thrumph;
Is to contrued veens may: my love other requint
me pioccause thou'n it.-
What. This simpely seasureful self trua,
By very will it; this thou, booked wrongs?
What infolttlest
```
