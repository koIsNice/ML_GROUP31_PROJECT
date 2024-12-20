import pandas as pd
import os 

irrelevant_cols = ['Id', 'Properties', 'Format', 'Time', 'Discrete', 'Realtime', 'Turns', 'Alternating', 'Simultaneous', 'HiddenInformation', 'Match', 'AsymmetricRules', 'AsymmetricPlayRules', 'AsymmetricEndRules', 'AsymmetricSetup', 'Players', 'NumPlayers', 'Simulation', 'Solitaire', 'TwoPlayer', 'Multiplayer', 'Coalition', 'Puzzle', 'DeductionPuzzle', 'PlanningPuzzle', 'Equipment', 'Container', 'Board', 'PrismShape', 'ParallelogramShape', 'RectanglePyramidalShape', 'TargetShape', 'BrickTiling', 'CelticTiling', 'QuadHexTiling', 'Hints', 'PlayableSites', 'Component', 'DiceD3', 'BiasedDice', 'Card', 'Domino', 'Rules', 'SituationalTurnKo', 'SituationalSuperko', 'InitialAmount', 'InitialPot', 'Play', 'BetDecision', 'BetDecisionFrequency', 'VoteDecisionFrequency', 'ChooseTrumpSuitDecision', 'ChooseTrumpSuitDecisionFrequency', 'LeapDecisionToFriend', 'LeapDecisionToFriendFrequency', 'HopDecisionEnemyToFriend', 'HopDecisionEnemyToFriendFrequency', 'HopDecisionFriendToFriend', 'FromToDecisionWithinBoard', 'FromToDecisionBetweenContainers', 'BetEffect', 'BetEffectFrequency', 'VoteEffectFrequency', 'SwapPlayersEffectFrequency', 'TakeControl', 'TakeControlFrequency', 'PassEffectFrequency', 'SetCost', 'SetCostFrequency', 'SetPhase', 'SetPhaseFrequency', 'SetTrumpSuit', 'SetTrumpSuitFrequency', 'StepEffectFrequency', 'SlideEffectFrequency', 'LeapEffectFrequency', 'HopEffectFrequency', 'FromToEffectFrequency', 'SwapPiecesEffect', 'SwapPiecesEffectFrequency', 'ShootEffect', 'ShootEffectFrequency', 'MaxCapture', 'OffDiagonalDirection', 'Information', 'HidePieceType', 'HidePieceOwner', 'HidePieceCount', 'HidePieceRotation', 'HidePieceValue', 'HidePieceState', 'InvisiblePiece', 'End', 'LineDrawFrequency', 'ConnectionDraw', 'ConnectionDrawFrequency', 'GroupLossFrequency', 'GroupDrawFrequency', 'LoopLossFrequency', 'LoopDraw', 'LoopDrawFrequency', 'PatternLoss', 'PatternLossFrequency', 'PatternDraw', 'PatternDrawFrequency', 'PathExtentEndFrequency', 'PathExtentWinFrequency', 'PathExtentLossFrequency', 'PathExtentDraw', 'PathExtentDrawFrequency', 'TerritoryLoss', 'TerritoryLossFrequency', 'TerritoryDraw', 'TerritoryDrawFrequency', 'CheckmateLoss', 'CheckmateLossFrequency', 'CheckmateDraw', 'CheckmateDrawFrequency', 'NoTargetPieceLoss', 'NoTargetPieceLossFrequency', 'NoTargetPieceDraw', 'NoTargetPieceDrawFrequency', 'NoOwnPiecesDraw', 'NoOwnPiecesDrawFrequency', 'FillLoss', 'FillLossFrequency', 'FillDraw', 'FillDrawFrequency', 'ScoringDrawFrequency', 'NoProgressWin', 'NoProgressWinFrequency', 'NoProgressLoss', 'NoProgressLossFrequency', 'SolvedEnd', 'Behaviour', 'StateRepetition', 'PositionalRepetition', 'SituationalRepetition', 'Duration', 'Complexity', 'BoardCoverage', 'GameOutcome', 'StateEvaluation', 'Clarity', 'Narrowness', 'Variance', 'Decisiveness', 'DecisivenessMoves', 'DecisivenessThreshold', 'LeadChange', 'Stability', 'Drama', 'DramaAverage', 'DramaMedian', 'DramaMaximum', 'DramaMinimum', 'DramaVariance', 'DramaChangeAverage', 'DramaChangeSign', 'DramaChangeLineBestFit', 'DramaChangeNumTimes', 'DramaMaxIncrease', 'DramaMaxDecrease', 'MoveEvaluation', 'MoveEvaluationAverage', 'MoveEvaluationMedian', 'MoveEvaluationMaximum', 'MoveEvaluationMinimum', 'MoveEvaluationVariance', 'MoveEvaluationChangeAverage', 'MoveEvaluationChangeSign', 'MoveEvaluationChangeLineBestFit', 'MoveEvaluationChangeNumTimes', 'MoveEvaluationMaxIncrease', 'MoveEvaluationMaxDecrease', 'StateEvaluationDifference', 'StateEvaluationDifferenceAverage', 'StateEvaluationDifferenceMedian', 'StateEvaluationDifferenceMaximum', 'StateEvaluationDifferenceMinimum', 'StateEvaluationDifferenceVariance', 'StateEvaluationDifferenceChangeAverage', 'StateEvaluationDifferenceChangeSign', 'StateEvaluationDifferenceChangeLineBestFit', 'StateEvaluationDifferenceChangeNumTimes', 'StateEvaluationDifferenceMaxIncrease', 'StateEvaluationDifferenceMaxDecrease', 'BoardSitesOccupied', 'BoardSitesOccupiedMinimum', 'BranchingFactor', 'BranchingFactorMinimum', 'DecisionFactor', 'DecisionFactorMinimum', 'MoveDistance', 'MoveDistanceMinimum', 'PieceNumber', 'PieceNumberMinimum', 'ScoreDifference', 'ScoreDifferenceMinimum', 'ScoreDifferenceChangeNumTimes', 'Roots', 'Cosine', 'Sine', 'Tangent', 'Exponential', 'Logarithm', 'ExclusiveDisjunction', 'Float', 'HandComponent', 'SetHidden', 'SetInvisible', 'SetHiddenCount', 'SetHiddenRotation', 'SetHiddenState', 'SetHiddenValue', 'SetHiddenWhat', 'SetHiddenWho']
game_cols = ['GameRulesetName', 'EnglishRules', 'LudRules']
output_cols = ['num_wins_agent1', 'num_draws_agent1', 'num_losses_agent1']
agent_cols = ['agent1', 'agent2']
dropped_cols = output_cols + irrelevant_cols + game_cols

def check_encoding_result(folder_path='dataset'):
    """
    讀取指定資料夾中的第一個CSV檔案並顯示完整的GameRulesetName欄位
    """
    try:
        # 讀取第一個CSV檔案
        first_file = "agent1_MCTS-ProgressiveHistory-0.1-MAST-false.csv"
        df = pd.read_csv(os.path.join(folder_path, first_file))
        
        # 找出所有GameRulesetName相關的欄位
        game_cols = [col for col in df.columns if 'GameRulesetName' in col]
        
        print("\nAll GameRulesetName columns and their values (first 5 rows):")
        print(df[game_cols].head())
        
        print("\nSum of each GameRulesetName column:")
        print(df[game_cols].sum())
            
    except Exception as e:
        print(f"Error checking game rules: {str(e)}")

def one_hot_encode_column(df, column_name):
    """
    對指定的列進行one-hot編碼
    """
    one_hot = pd.get_dummies(df[column_name], prefix=column_name, dtype=int)
    df = pd.concat([df, one_hot], axis=1)
    df = df.drop(columns=[column_name])
    return df

def save_agent_data(df, agent_col):

    split_values = df[agent_col].copy().str.split('-')
    max_parts = len(split_values[0])


    for i in range(1, max_parts):
        col_name = f"{agent_col}_part{i+1}"

        l = []
        for j in range(len(split_values)):
            l.append(split_values[j][i])

        df[col_name] = l
        df = one_hot_encode_column(df, col_name)
    df = df.drop(columns=[agent_col])

    return df

    # agent_types = df[agent_col].unique()
    
    # for agent in agent_types:
    #     agent_data = df[df[agent_col] == agent].copy()
        
    #     # 對這個agent的資料做split和one-hot encoding
    #     split_values = pd.Series([agent]).str.split('-')[0]
    #     print(split_values)
    #     break
    #     max_parts = len(split_values)
        
    #     for i in range(max_parts):
    #         col_name = f"{agent_col}_part{i+1}"
    #         agent_data[col_name] = split_values[i]
    #         agent_data = one_hot_encode_column(agent_data, col_name)
        
    #     clean_agent_name = agent
    #     filename = f"{agent_col}_{clean_agent_name}.csv"
    #     filepath = os.path.join(output_foldername, filename)
    #     agent_data = agent_data.drop(columns=[agent_col])
    #     agent_data.to_csv(filepath, index=False)
    #     print(f"Saved: {filepath}")

def process_and_save_csv(df):
    #output_foldername = 'dataset_preprocess'
    output_foldername = 'dataset_preprocess_submit'
    os.makedirs(output_foldername, exist_ok=True)
    
        #df = pd.read_csv('dataset/train.csv')

    for agent_col in ['agent1', 'agent2']:
        if agent_col in df.columns:
            df = save_agent_data(df, agent_col)
        else:
            raise KeyError(f"Missing required column: '{agent_col}'")

    df = one_hot_encode_column(df, 'GameRulesetName')

    df = df.drop(columns=dropped_cols, errors='ignore')

    #print(df.head())
    
    filename = "preprocessed_dataset.csv"
    filepath = os.path.join(output_foldername, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")      
    
    return df

def process_for_test(df):

    #df = one_hot_encode_column(df, 'GameRulesetName')
    cols = pd.read_csv('dataset_preprocess_submit/cols.csv')
    df2 = pd.DataFrame(0, index=range(len(df)), columns=cols.columns)
    #print(df2.shape)

    for idx, col in enumerate(cols):
        for i in range(len(df)):
            if col[:6] == 'agent1' and col.split('_')[-1] in str(df.iloc[i, 2]).split('-'):
                df2[col].iloc[i] = 1
            elif col[:6] == 'agent2' and col.split('_')[-1] in str(df.iloc[i, 3]).split('-'):
                df2[col].iloc[i] = 1
            elif col.split('_')[0] == 'GameRulesetName' and col.split('_', 1)[1] == str(df.iloc[i, 1]):
                df2[col].iloc[i] = 1
                
    df = df.drop(columns=dropped_cols + agent_cols, errors='ignore')
    df = pd.concat([df, df2], axis=1)
    return df

import pickle
if __name__ == "__main__":

    # output_foldername = 'dataset_preprocess'
    # os.makedirs(output_foldername, exist_ok=True)

    # df = pd.read_csv('dataset_preprocess_submit/cols2.csv')
    # print(df.columns[613])
    # df = process_for_test(df)
    # df.to_csv('dataset_preprocess_submit/test.csv', index=False)
    # print(df.shape)
    # process_and_save_csv(df)
    #check_encoding_result()
    df = pd.read_csv('dataset/train.csv')
    print(df[['agent1', 'agent2', 'utility_agent1']])
