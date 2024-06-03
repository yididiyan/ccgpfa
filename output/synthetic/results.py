import pickle
import glob

# print('Results with inducing points')


for llk in ['binom', 'negbinom']:
    for m in [None, 50, 100, 200]:

        print(m, llk, end=' & ')
        for t in [300, 900, 1500]:
            matches = glob.glob(f'data_10_100_{t}/{llk}_*n_inducing{m}*/summary.pkl')

            if matches:
                data = pickle.load(open(matches[0], 'rb'))
                print('{0:.3f}'.format(data['train']['test_llk_mean']),'$\pm$ {0:.3f}'.format(data['train']['test_llk_std']), '({0:.2f})'.format(data['train']['elapsed_time']), end=' & ' )
        print()