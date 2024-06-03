import pickle
import glob


for t in [300, 900, 1500]:
    matches = glob.glob(f'./results_T__{t}__0_1000_*.pkl')
    if matches:
        data = pickle.load(open(matches[0], 'rb'))
        print('{0:.3f}'.format(data['test_loglik_mean']),  '{0:.3f}'.format(data['test_loglik_std'], data['time_taken']))
print()