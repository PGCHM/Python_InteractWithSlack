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
upload predictions
'''
# from matplotlib import MatplotlibDeprecationWarning
# warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning)


def plot_image(image, ax=None):
    if ax is not None:
        ax.imshow(image.reshape((28, 28)), cmap='Greys')
    else:
        plt.imshow(image.reshape((28, 28)), cmap='Greys')

plot_image(x_test[10])
print(y_test[1])

import os

def plot_predictions(n=4):
    """Plot test image and predictions"""

    # Get random images to plot
    to_plot = np.random.choice(list(range(x_test.shape[0])),
                               size=n, replace=False)
    correct = []
    # Make predictions and plot each image
    for i in to_plot:
        image = x_test[i]
        probs = model.predict_proba(image.reshape((1, 28, 28, 1)))[0]
        pred = pd.DataFrame({'prob': probs})
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        plot_image(image, axs[0])

        pred['prob'].plot.bar(ax=axs[1])
        axs[1].set_xlabel('Class')
        axs[1].set_ylabel('Probability')
        axs[1].set_title('Predictions')
        plt.savefig(os.path.join(r'C:\temp\images', f'test-{i}-predictions.png'))
        plt.show()

        correct.append(np.argmax(probs) == np.argmax(y_test[i]))

    return to_plot, correct


to_plot, c = plot_predictions()


def post_predictions(channel, n=4):
    """Post Keras preditions to Slack"""

    # Make predictions
    plot_indexes, correct = plot_predictions(n=n)

    # Iterate through images and correct indicators
    for i, r in zip(plot_indexes, correct):
        filename = os.path.join(r'C:\temp\images', f'test-{i}-predictions.png')
        # Upload the files
        r = slack.files.upload(file_=filename,
                               title="Predictions",
                               channels=[channel],
                               initial_comment='Correct' if r else 'Incorrect')


post_predictions(reporting_channel)

