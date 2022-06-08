print(f'Hello from {__file__}')

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# downloading from https://www.kaggle.com/competitions/sf-matml-2022-classification/data
# there is one file .zip 
# we write to the current directory with './'
api.competition_download_file('sf-matml-2022-classification', 'Target.csv', path='./data/raw/train/')
api.competition_download_file('sf-matml-2022-classification', 'Train.csv', path='./data/raw/train/')
api.competition_download_file('sf-matml-2022-classification', 'Test.csv', path='./data/raw/test/')
api.competition_download_file('sf-matml-2022-classification', 'Submission.csv', path='./data/raw/test/')
