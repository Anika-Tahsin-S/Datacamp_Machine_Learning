##                  Bayes Rule in Python                  ##
# Part 1
# Assign probabilities to variables 
p_unhappy = 0.15
p_unhappy_close = 0.35

# Part 2
# Probabiliy someone will close
p_close = 0.07

# Part 3
# Probability unhappy person will close
p_close_unhappy = (p_unhappy_close * p_close) / p_unhappy
print(p_close_unhappy)

# output: 0.16333333333333336






##                  Bayesian Hyperparameter tuning with Hyperopt                  ##
import hyperopt as hp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# Set up space dictionary with specified hyperparamters
space = {'max_depth': hp.quniform('max_depth', 2, 10, 2),
         'learning_rate': hp.uniform('learning_rate', 0.001, 0.9)}

# Set up objective function
def objective(params):
    params = {'max_depth': int(params['max_depth']), 
              'learning_rate': params['learning_rate']}
    gbm_clf = GradientBoostingClassifier(n_estimators=100, **params)
    best_score = cross_val_score(gbm_clf, X_train, y_train, 
                                 scoring='accuracy', cv=2, n_jobs=4).mean()
    loss = 1 - best_score
    return loss

# Run the algorithm
best = fmin(fn=objective, space=space, max_evals=20, 
               rstate=np.random.RandomState(42), algo= tpe.suggest)
print(best)

# output:
    
  0%|          | 0/20 [00:00<?, ?it/s, best loss: ?]
  5%|5         | 1/20 [00:00<00:05,  3.35it/s, best loss: 0.26759418985474637]
 10%|#         | 2/20 [00:00<00:06,  2.96it/s, best loss: 0.2549063726593165] 
 15%|#5        | 3/20 [00:00<00:05,  3.33it/s, best loss: 0.2549063726593165]
 20%|##        | 4/20 [00:01<00:04,  3.86it/s, best loss: 0.2549063726593165]
 25%|##5       | 5/20 [00:01<00:04,  3.10it/s, best loss: 0.2549063726593165]
 30%|###       | 6/20 [00:01<00:04,  3.18it/s, best loss: 0.2549063726593165]
 35%|###5      | 7/20 [00:02<00:03,  3.68it/s, best loss: 0.2549063726593165]
 40%|####      | 8/20 [00:02<00:03,  3.98it/s, best loss: 0.2549063726593165]
 45%|####5     | 9/20 [00:02<00:02,  4.21it/s, best loss: 0.2549063726593165]
 50%|#####     | 10/20 [00:02<00:02,  4.49it/s, best loss: 0.2549063726593165]
 55%|#####5    | 11/20 [00:02<00:01,  4.83it/s, best loss: 0.2549063726593165]
 60%|######    | 12/20 [00:03<00:01,  4.59it/s, best loss: 0.2549063726593165]
 65%|######5   | 13/20 [00:03<00:01,  4.32it/s, best loss: 0.2549063726593165]
 70%|#######   | 14/20 [00:03<00:02,  2.82it/s, best loss: 0.2525688142203555]
 75%|#######5  | 15/20 [00:04<00:01,  3.00it/s, best loss: 0.2525688142203555]
 80%|########  | 16/20 [00:04<00:01,  3.25it/s, best loss: 0.2525688142203555]
 85%|########5 | 17/20 [00:05<00:01,  2.59it/s, best loss: 0.24246856171404285]
 90%|######### | 18/20 [00:05<00:00,  2.96it/s, best loss: 0.24246856171404285]
 95%|#########5| 19/20 [00:05<00:00,  3.37it/s, best loss: 0.24246856171404285]
100%|##########| 20/20 [00:05<00:00,  3.79it/s, best loss: 0.24246856171404285]
100%|##########| 20/20 [00:05<00:00,  3.53it/s, best loss: 0.24246856171404285]
    {'learning_rate': 0.11310589268581149, 'max_depth': 6.0}






