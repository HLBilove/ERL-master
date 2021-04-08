# ERL

## Setups
The requiring environment is as bellow:  

- Linux 
- Python 3+
- PyTorch 1.2.0 
- Scikit-learn 0.21.3
- Scipy 1.3.1
- Numpy 1.17.2

## Notes

- 's' and 'e' in code denote skill and exercise, respectively.
- 't' and 'f' in code denote the ture and false response count in the past, respectively.
- 'sd' and 'rd' in code denote the sequence delay and repeat delay in minute, respectively.
- 'x' and 'y' in code denote two side factors.

## Running ERL.
Here are some examples for using ERL model based AKT:

(BE without skill)
```
python main.py --dataset assist2009 --model akt
python main.py --dataset assist2017 --model akt
```

(BE)
```
python main.py --dataset assist2009 --model akt_e
python main.py --dataset assist2017 --model akt_e
```

(BE+PE)
```
python main.py --dataset assist2009 --model akt_e_p
python mai.py --dataset assist2017 --model akt_e_p
```

(BE+PE+FE)
```
python main.py --dataset assist2017 --model akt_e_p_f
python mai.py --dataset statics --model akt_e_p_f
```

(ERL)
```
python main.py --dataset assist2017 --model akt_e_p_f_a
python mai.py --dataset statics --model akt_e_p_f_a
```
