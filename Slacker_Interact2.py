'''
Acknowledgements:
Ideas coming from Will K.'s post
https://github.com/WillKoehrsen/Data-Analysis/tree/master/slack_interaction
'''

# ALl outputs 
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# Slacker
from slacker import Slacker

# Data manipulation
import numpy as np
import pandas as pd

# %load_ext autoreload
# %autoreload 2

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('fivethirtyeight')

# Displaying images
from IPython.display import Image

with open(r"C:\temp\slack_api_python.txt", 'r') as f:
    slack_api_token = f.read()

# Connect to Slack
slack = Slacker(slack_api_token)
if slack.api.test().successful:
    print(
        f"Successfully connected to {slack.team.info().body['team']['name']}.")
else:
    print('Try Again!')

'''
Create channel
'''
channel_list = [c['name'] for c in slack.channels.list().body['channels']]

reporting_channel = 'tf_training_report'

if reporting_channel not in channel_list:
    slack.channels.create('tf_training_report')

cid = slack.channels.get_channel_id('tf_training_report')
r = slack.channels.set_purpose(
    cid, 'Report progress while training machine learning models')
r = slack.channels.set_topic(cid, 'Progress Monitoring')
r = slack.channels.info(cid).body
r['channel']['latest']


'''
Model
'''
from utils import get_data_and_model

x_train, x_test, y_train, y_test, model = get_data_and_model()
model.summary()


'''
Call back
'''
from keras.callbacks import Callback
from datetime import datetime


def report_stats(text, channel):
    """Report training stats"""
    r = slack.chat.post_message(channel=channel, text=text,
                                username='Training Report',
                                icon_emoji=':clipboard:')

    if r.successful:
        return True
    else:
        return r.error


from timeit import default_timer as timer


class SlackUpdate(Callback):
    """Custom Keras callback that posts to Slack while training a neural network"""

    def __init__(self, channel):
        self.channel = channel

    def on_train_begin(self, logs={}):
        report_stats(text=f'Training started at {datetime.now()}',
                     channel=reporting_channel)

        self.start_time = timer()
        self.train_acc = []
        self.valid_acc = []
        self.train_loss = []
        self.valid_loss = []
        self.n_epochs = 0

    def on_epoch_end(self, batch, logs={}):

        self.train_acc.append(logs.get('acc'))
        self.valid_acc.append(logs.get('val_acc'))
        self.train_loss.append(logs.get('loss'))
        self.valid_loss.append(logs.get('val_loss'))
        self.n_epochs += 1

        message = f'Epoch: {self.n_epochs} Training Loss: {self.train_loss[-1]:.4f} Validation Loss: {self.valid_loss[-1]:.4f}'

        report_stats(message, channel=self.channel)

    def on_train_end(self, logs={}):

        best_epoch = np.argmin(self.valid_loss)
        valid_loss = self.valid_loss[best_epoch]
        train_loss = self.train_loss[best_epoch]
        train_acc = self.train_acc[best_epoch]
        valid_acc = self.valid_acc[best_epoch]

        message = f'Trained for {self.n_epochs} epochs. Best epoch was {best_epoch + 1}.'
        report_stats(message, channel=self.channel)
        message = f'Best validation loss = {valid_loss:.4f} Training Loss = {train_loss:.2f} Validation accuracy = {100*valid_acc:.2f}%'
        report_stats(message, channel=self.channel)

'''
real slack update while model being run
'''
import tensorflow as tf
updater = SlackUpdate(channel=reporting_channel)


# Fit the model for 10 epochs
history = model.fit(x_train, y_train, epochs=12, batch_size=512,
                    callbacks=[updater], validation_split=0.4)


'''
plot results
'''
from utils import plot_history

plot_history(history.history)
plt.savefig('training_curves.png')


'''
helpful comments added
'''
# Find minimum loss
min_loss = min(history.history['val_loss'])
best_epoch = np.argmin(history.history['val_loss']) + 1

# Upload file
comment = f"Best loss of {min_loss:.4f} at epoch {best_epoch}."
r = slack.files.upload(file_='training_curves.png', title="Training Curves", channels=[reporting_channel],
                       initial_comment=comment)
r = slack.channels.info(cid).body
r['channel']['latest']['text']
