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