## ccGPFA 
The repository contains an implementation for "Conditionally-Conjugate Gaussian Process Factor Analysis for Spike Count Data via Data Augmentation" (To appear at ICML 2024). 



## Requirements 

* Python 3.8.10. Run the following script to setup the environment.

```
conda create -n ccgpfa-py38 python=3.8.10
conda activate ccgpfa-py38
pip install -r requirements.txt 
```


For the PAL baseline from Keeley et. al 2020, we create a separate environment(compatible with an older version). We include the PAL implementation under `Count_GPFA`.
```
cd Count_GPFA
conda create -n PAL-py356 python=3.5.6
conda activate PAL-py356
pip install -r requirements.txt
```



## Experiments

### Experiment 1: Allen Observatory Visual Coding Neuropixels 

We include the dataset preprocessed from Allen Observatory data under `datasets/drifting_gratings_75_repeats/`. The dataset constains 75 trials of V1 recording under drifting gratings stimulus with 45 degress and contrast of 0.1.

####  Make output directory & switch to experiments 
```
mkdir ../output 
cd experiments/multiple_trials
```


#### Running ccGPFA(negative binomial) on the allen data   
```
python run_with_behavior_mapping.py --dataset ../../datasets/drifting_gratings_75_repeats/data_15_45.0_0.1_V1.pkl --output_dir ../output/drifting_gratings_75_repeats --test-size 25 --n-iter 100 --llk negbinom --ell 5. 
```

#### Running ccGPFA(binomial) on the allen data   
```
python run_with_behavior_mapping.py --dataset ../../datasets/drifting_gratings_75_repeats/data_15_45.0_0.1_V1.pkl --output_dir ../output/drifting_gratings_75_repeats --test-size 25 --n-iter 100 --llk binom --ell 5. 
```


#### Running bGPFA method (Jensen et al. 2021 ) 
( using its official implementation from [[Github](https://github.com/tachukao/mgplvm-pytorch)])
``` 
python run_bgpfa.py --dataset ../../datasets/drifting_gratings_75_repeats/data_15_45.0_0.1_V1.pkl --output_dir ../../output/drifting_gratings_75_repeats --test-size 25 --n-iter 1000 --llk negbinom --ell 5.
```

#### Running PAL method (Keeley et. al 2020)

For PAL, we used the publicly available implementation on [[Github](https://github.com/skeeley/Count_GPFA)]
```
cd Count_GPFA
python PAL_public_allen.py --dataset ../datasets/drifting_gratings_75_repeats/data_15_45.0_0.1_V1.pkl --max-iter 150
```

For GPFA (Yu et al. 2008), we used Elephant library, sample code included under 
```
 python run_gpfa.py --dataset ../../datasets/drifting_gratings_75_repeats/data_15_45.0_0.1_V1.pkl --output_dir ../../output/drifting_gratings_75_repeats --test-size 25
```


### Experiment 2: MC_Maze Reaching Task  
A subset of reaching task data is preprocessed from the MC_Maze_L data. In total, 9 conditions (with varying angles) are included under `datasets/MC_Maze_L`.  


```
cd experiments/multiple_trials
```


```
python run_with_behavior_mapping.py --dataset ../../datasets/MC_Maze_L/data_5_cond${i}_0.pkl --output_dir ../../output/MC_Maze_L --n-iter 500 --llk binom --ell 20. --threshold 0.01   --kinematic True
```
Run the above script for all files in `datasets/MC_Maze_L` by replacing `${i}` by the condition number. For negative binomial ccGPFA model switch `binom` to `negbinom`.

### Running bGPFA 
```
python run_bgpfa_with_behavior_mapping.py --dataset ../../datasets/MC_Maze_L/data_5_cond${i}_0.pkl --output_dir ../../output/MC_Maze_L --n-iter 2000 --llk negbinom --ell 20.   --kinematic True
```

Run the above script for all files in `datasets/MC_Maze_L` by replacing `${i}` by the condition number. 


### Running PAL 
After switching to the PAL environment using `conda activate PAL-py356`,  

```
cd Count_GPFA 
python PAL_public_MC_Maze.py --dataset ../datasets/MC_Maze_L/<pkl-file>
```

Final analysis can be found at `(experiments/multiple_trials/MC Maze - Large - Analysis .ipynb)`

### Running GPFA 
```
python run_gpfa.py --dataset ../datasets/MC_Maze_L/<pkl-file> --output_dir ../../output/MC_Maze_L --test-size 0 --bin-size 15 --threshold 1
```



## Scalability Experiments 

#### T = 1500 
```
for i in 50 100 200; do python run_with_behavior_mapping.py --dataset  ../../datasets/synthetic/data_10_100_1500.pkl --output ../../output/synthetic/ --test-size 3  --n-iter 1500 --llk negbinom --ell 20. --threshold 0.1 --n-inducing $i  --lr 0.25 --batch 200; done;
```

#### T = 900 
```
for i in 50 100 200; do python run_with_behavior_mapping.py --dataset  ../../datasets/synthetic/data_10_100_900.pkl --output ../../output/synthetic/ --test-size 3 --n-iter 1500 --llk negbinom --ell 20. --threshold 0.1 --n-inducing $i  --lr 0.25 --batch 150; done;
```

#### T= 300
```
for i in 50 100 200; do python run_with_behavior_mapping.py --dataset  ../../datasets/synthetic/data_10_100_300.pkl --output ../../output/synthetic/ --test-size 3 --n-iter 1500 --llk negbinom --ell 20. --threshold 0.1 --n-inducing $i  --lr 0.25 --batch 100; done; 
```


Switch `negbinom` to `binom` for equivalent Binomial ccGPFA.  


### Without inducing points 
```
python run_with_behavior_mapping.py --dataset  ../../datasets/synthetic/data_10_100_1500.pkl --output ../../output/synthetic/ --test-size 3 --n-iter 1500 --llk binom --ell 20. --threshold 0.1
python run_with_behavior_mapping.py --dataset  ../../datasets/synthetic/data_10_100_1500.pkl --output ../../output/synthetic/ --test-size 3 --n-iter 1500 --llk negbinom --ell 20. --threshold 0.1
```


### PAL 
For PAL, we switch to its environment and execute   
```
cd Count_GPFA
python PAL_public_synthetic.py --file ../datasets/synthetic/data_10_100_300.pkl --max-iter 1000
python PAL_public_synthetic.py --file ../datasets/synthetic/data_10_100_900.pkl --max-iter 1000
python PAL_public_synthetic.py --file ../datasets/synthetic/data_10_100_1500.pkl --max-iter 1000
```