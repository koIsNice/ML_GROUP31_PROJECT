import os
import sys
import warnings
import pickle
warnings.filterwarnings('ignore')

import numpy as np
import polars as pl
from sklearn.model_selection import KFold
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import catboost

import kaggle_evaluation.mcts_inference_server
from preprocess import *

irrelevant_cols = ['Id', 'Properties', 'Format', 'Time', 'Discrete', 'Realtime', 'Turns', 'Alternating', 'Simultaneous', 'HiddenInformation', 'Match', 'AsymmetricRules', 'AsymmetricPlayRules', 'AsymmetricEndRules', 'AsymmetricSetup', 'Players', 'NumPlayers', 'Simulation', 'Solitaire', 'TwoPlayer', 'Multiplayer', 'Coalition', 'Puzzle', 'DeductionPuzzle', 'PlanningPuzzle', 'Equipment', 'Container', 'Board', 'PrismShape', 'ParallelogramShape', 'RectanglePyramidalShape', 'TargetShape', 'BrickTiling', 'CelticTiling', 'QuadHexTiling', 'Hints', 'PlayableSites', 'Component', 'DiceD3', 'BiasedDice', 'Card', 'Domino', 'Rules', 'SituationalTurnKo', 'SituationalSuperko', 'InitialAmount', 'InitialPot', 'Play', 'BetDecision', 'BetDecisionFrequency', 'VoteDecisionFrequency', 'ChooseTrumpSuitDecision', 'ChooseTrumpSuitDecisionFrequency', 'LeapDecisionToFriend', 'LeapDecisionToFriendFrequency', 'HopDecisionEnemyToFriend', 'HopDecisionEnemyToFriendFrequency', 'HopDecisionFriendToFriend', 'FromToDecisionWithinBoard', 'FromToDecisionBetweenContainers', 'BetEffect', 'BetEffectFrequency', 'VoteEffectFrequency', 'SwapPlayersEffectFrequency', 'TakeControl', 'TakeControlFrequency', 'PassEffectFrequency', 'SetCost', 'SetCostFrequency', 'SetPhase', 'SetPhaseFrequency', 'SetTrumpSuit', 'SetTrumpSuitFrequency', 'StepEffectFrequency', 'SlideEffectFrequency', 'LeapEffectFrequency', 'HopEffectFrequency', 'FromToEffectFrequency', 'SwapPiecesEffect', 'SwapPiecesEffectFrequency', 'ShootEffect', 'ShootEffectFrequency', 'MaxCapture', 'OffDiagonalDirection', 'Information', 'HidePieceType', 'HidePieceOwner', 'HidePieceCount', 'HidePieceRotation', 'HidePieceValue', 'HidePieceState', 'InvisiblePiece', 'End', 'LineDrawFrequency', 'ConnectionDraw', 'ConnectionDrawFrequency', 'GroupLossFrequency', 'GroupDrawFrequency', 'LoopLossFrequency', 'LoopDraw', 'LoopDrawFrequency', 'PatternLoss', 'PatternLossFrequency', 'PatternDraw', 'PatternDrawFrequency', 'PathExtentEndFrequency', 'PathExtentWinFrequency', 'PathExtentLossFrequency', 'PathExtentDraw', 'PathExtentDrawFrequency', 'TerritoryLoss', 'TerritoryLossFrequency', 'TerritoryDraw', 'TerritoryDrawFrequency', 'CheckmateLoss', 'CheckmateLossFrequency', 'CheckmateDraw', 'CheckmateDrawFrequency', 'NoTargetPieceLoss', 'NoTargetPieceLossFrequency', 'NoTargetPieceDraw', 'NoTargetPieceDrawFrequency', 'NoOwnPiecesDraw', 'NoOwnPiecesDrawFrequency', 'FillLoss', 'FillLossFrequency', 'FillDraw', 'FillDrawFrequency', 'ScoringDrawFrequency', 'NoProgressWin', 'NoProgressWinFrequency', 'NoProgressLoss', 'NoProgressLossFrequency', 'SolvedEnd', 'Behaviour', 'StateRepetition', 'PositionalRepetition', 'SituationalRepetition', 'Duration', 'Complexity', 'BoardCoverage', 'GameOutcome', 'StateEvaluation', 'Clarity', 'Narrowness', 'Variance', 'Decisiveness', 'DecisivenessMoves', 'DecisivenessThreshold', 'LeadChange', 'Stability', 'Drama', 'DramaAverage', 'DramaMedian', 'DramaMaximum', 'DramaMinimum', 'DramaVariance', 'DramaChangeAverage', 'DramaChangeSign', 'DramaChangeLineBestFit', 'DramaChangeNumTimes', 'DramaMaxIncrease', 'DramaMaxDecrease', 'MoveEvaluation', 'MoveEvaluationAverage', 'MoveEvaluationMedian', 'MoveEvaluationMaximum', 'MoveEvaluationMinimum', 'MoveEvaluationVariance', 'MoveEvaluationChangeAverage', 'MoveEvaluationChangeSign', 'MoveEvaluationChangeLineBestFit', 'MoveEvaluationChangeNumTimes', 'MoveEvaluationMaxIncrease', 'MoveEvaluationMaxDecrease', 'StateEvaluationDifference', 'StateEvaluationDifferenceAverage', 'StateEvaluationDifferenceMedian', 'StateEvaluationDifferenceMaximum', 'StateEvaluationDifferenceMinimum', 'StateEvaluationDifferenceVariance', 'StateEvaluationDifferenceChangeAverage', 'StateEvaluationDifferenceChangeSign', 'StateEvaluationDifferenceChangeLineBestFit', 'StateEvaluationDifferenceChangeNumTimes', 'StateEvaluationDifferenceMaxIncrease', 'StateEvaluationDifferenceMaxDecrease', 'BoardSitesOccupied', 'BoardSitesOccupiedMinimum', 'BranchingFactor', 'BranchingFactorMinimum', 'DecisionFactor', 'DecisionFactorMinimum', 'MoveDistance', 'MoveDistanceMinimum', 'PieceNumber', 'PieceNumberMinimum', 'ScoreDifference', 'ScoreDifferenceMinimum', 'ScoreDifferenceChangeNumTimes', 'Roots', 'Cosine', 'Sine', 'Tangent', 'Exponential', 'Logarithm', 'ExclusiveDisjunction', 'Float', 'HandComponent', 'SetHidden', 'SetInvisible', 'SetHiddenCount', 'SetHiddenRotation', 'SetHiddenState', 'SetHiddenValue', 'SetHiddenWhat', 'SetHiddenWho']
game_cols = ['GameRulesetName', 'EnglishRules', 'LudRules']
output_cols = ['num_wins_agent1', 'num_draws_agent1', 'num_losses_agent1']
agent_cols = ['agent1', 'agent2']
dropped_cols = output_cols + irrelevant_cols + game_cols


class Config:
    #train_path = 'dataset/train.csv'
    train_path = 'dataset_preprocess/preprocessed_dataset_double_2.csv'
    
    early_stop = 50
    n_splits = 5
    seed = 1212
    split_agent_features = True
    
    lgbm_params = {
        'num_boost_round': 10_000,
        'seed': 1212,
        'verbose': -1,
        'num_leaves': 63,
        'learning_rate': 0.05,
        'max_depth': 8,
        'reg_lambda': 1.0,
    }

    cat_params1={'task_type'           : "GPU",
                'eval_metric'         : "RMSE",
                'bagging_temperature' : 0.50,
                'iterations'          : 3096,
                'learning_rate'       : 0.08,
                'max_depth'           : 12,
                'l2_leaf_reg'         : 1.25,
                'min_data_in_leaf'    : 24,
                'random_strength'     : 0.25, 
                'verbose'             : 1,
                }
    
    cat_params2={'task_type'           : "GPU",
            'eval_metric'         : "RMSE",
            'bagging_temperature' : 0.60,
            'iterations'          : 3096,
            'learning_rate'       : 0.08,
            'max_depth'           : 12,
            'l2_leaf_reg'         : 1.25,
            'min_data_in_leaf'    : 24,
            'random_strength'     : 0.20, 
            'max_bin'             :2048,
            'verbose'             : 1,
            }
    

def delete_less_freq_col(df, mode='train'):
    column_sums = df.sum(axis=0)
    if mode == 'train':
        column_sums = column_sums.iloc[613:] # only consider the cols with game rule
    else:
        column_sums = column_sums.iloc[612:]

    avg = column_sums.sum()/len(column_sums)
    var = column_sums.var()
    lower_bound = avg-var**(1/2)
    #print('lower bound: {}'.format(lower_bound))

    to_delete_df = column_sums[column_sums < lower_bound]

    selected_cols = to_delete_df.index.to_list()

    processed_df = df.drop(columns=selected_cols)
    return selected_cols, processed_df


def process_data(df): 
    df = df.drop(filter(lambda x: x in df.columns, dropped_cols))
    if Config.split_agent_features:
        for col in agent_cols:
            df = df.with_columns(pl.col(col).str.split(by="-").list.to_struct(fields=lambda idx: f"{col}_{idx}")).unnest(col).drop(f"{col}_0")
    df = df.with_columns([pl.col(col).cast(pl.Categorical) for col in df.columns if col[:6] in agent_cols])            
    df = df.with_columns([pl.col(col).cast(pl.Float32) for col in df.columns if col[:6] not in agent_cols])
    print(f'Data shape: {df.shape}')
    return df.to_pandas()


def train_lgb(data):
    X = data.drop(['utility_agent1'], axis=1)
    y = data['utility_agent1']

    cv = KFold(n_splits=Config.n_splits, shuffle=True, random_state=Config.seed)
    models = []
    for fi, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        print(f'Fold {fi+1}/{Config.n_splits} ...')
        # model = lgb.LGBMRegressor(**Config.lgbm_params)
        # model.fit(X.iloc[train_idx], y.iloc[train_idx],
        #           eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])],
        #           eval_metric='rmse',
        #           callbacks=[lgb.early_stopping(Config.early_stop)])
        model = catboost.CatBoostRegressor(**Config.cat_params2)
        model.fit(X.iloc[train_idx], y.iloc[train_idx],
                  eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])])
        models.append(model)

    with open('models/cat_deleted_2_double.pkl','wb') as f:
        pickle.dump(models,f)

    return models

def infer_lgb(data, models):
    return np.mean([model.predict(data) for model in models], axis=0)


run_i = 0
def predict(test_data, submission):
    #print(process_and_save_csv(test_data.to_pandas()).head())
    #print(process_data(test_data).head())
    #print(test_data.to_pandas().columns[-1])

    global run_i, models
    if run_i == 0:
        train_df = pl.read_csv(Config.train_path)
        #models = train_lgb(process_data(train_df))
        #models = train_lgb(train_df.to_pandas())
        # _, train_data = delete_less_freq_col(train_df.to_pandas())
        train_data = train_df.to_pandas().drop(columns=col_to_be_dropped)
        models = train_lgb(train_data)
    run_i += 1
    
    #test_data = process_data(test_data)
    #test_data = process_for_test(test_data.to_pandas())
    test_data = process_for_test(test_data.to_pandas())
    test_data = test_data.drop(columns=col_to_be_dropped)
    return submission.with_columns(pl.Series('utility_agent1', infer_lgb(test_data, models)))

train_df = pl.read_csv('dataset_preprocess/preprocessed_dataset.csv').to_pandas()
col_to_be_dropped, deleted_dff = delete_less_freq_col(train_df)

with open('dataset_preprocess_submit/col_to_be_dropped.pkl','wb') as f:
    pickle.dump(col_to_be_dropped,f)

print('col saved')

# inference_server = kaggle_evaluation.mcts_inference_server.MCTSInferenceServer(predict)
# if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
#     inference_server.serve()
# else:
#     inference_server.run_local_gateway(
#         (
#             'dataset/test.csv',
#             'dataset/sample_submission.csv'
#         )
#     )
