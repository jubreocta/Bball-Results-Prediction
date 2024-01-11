# Bball-Results-Prediction

This app predicts the outcome of a given set of games given a history of games played.
The app is built on the nba data pulled from basketball reference https://www.basketball-reference.com/ from the 2004 - 2005 nba season when the league went to 82 teams.
The app is built to ingest any kind of basketball data but may use keywords native to the NBA.

The main file is the Compile.py script in the root folder which runs everything.

This script has been built in such a way that everyone of the step itemized below is a one liner in the Compile.py Script.

Some of these step can be commented out if they would produce redundant actions. e.g publishing a .csv file to a folder can be commented out if the file already exist in the said folder.

Steps

1.  The raw data of game results is put in a folder labeled "raw".

2.  A load connector pulls the data from 'raw',transform the data into the formnat specified below and places a compiled 'RawData.csv' file in the Data folder.

Format is columns:
    'Season', 'Date', 'LeftTeam', 'LeftScore', 'RightScore', 'RightTeam', 'Overtime' in that specific order.
All basketball data needs to be in this format to progress using the functions.

3.  Ranking algorithms act on RawData.csv to produce features to be used in predictions. The features generated are added to our data set and saved as another file FinalData.csv.

Features are:
    'Left_C_1', 'Right_C_1',
    'Left_MV1_1', 'Right_MV1_1',
    'Left_MV2_1', 'Right_MV2_1',
    'Left_MV3_1', 'Right_MV3_1', 
    'Left_MO_1', 'Right_MO_1',
    'Left_MD_1', 'Right_MD_1',
    'Left_M_1', 'Right_M_1',
    'Left_ODO_1', 'Right_ODO_1',
    'Left_ODD_1', 'Right_ODD_1',
    'Left_OD_1', 'Right_OD_1',
    'Left_WP_1', 'Right_WP_1',
    'Left_C_2', 'Right_C_2',
    'Left_MV1_2', 'Right_MV1_2',
    'Left_MV2_2', 'Right_MV2_2',
    'Left_MV3_2', 'Right_MV3_2',
    'Left_MO_2', 'Right_MO_2',
    'Left_MD_2', 'Right_MD_2',
    'Left_M_2', 'Right_M_2',
    'Left_ODO_2', 'Right_ODO_2',
    'Left_ODD_2', 'Right_ODD_2',
    'Left_OD_2', 'Right_OD_2',
    'Left_WP_2', 'Right_WP_2',
    'Left_B2B', 'Right_B2B',
    'Left_EWMA_7', 'Right_EWMA_7',
    'Left_EWMA_28', 'Right_EWMA_28'
50 total