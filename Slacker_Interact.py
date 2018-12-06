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
r = slack.team.info()
print(r.body)

from IPython.display import Image
Image(url='https://avatars.slack-edge.com/2018-11-30/492734348690_802c4805ab4a0d29383b_230.png')

from utils import get_data_and_model, get_options
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

command_dict = get_options(slack)
print(command_dict['functions']['users'])
print(command_dict['functions']['channels'])


channels = slack.channels.list().body

# Iterate through channels
for channel in channels['channels']:
    print(
        f'Channel {channel["name"]} Purpose: {channel["purpose"]["value"]}\n')
'''

users = slack.users.list().body
'''
for user in users['members']:
    # Print some information
    print(
        f'\nUser: {user["name"]}, Real Name: {user["real_name"]}, Time Zone: {user["tz_label"]}.')
    print(f'Current Status: {user["profile"]["status_text"]}')
    # Get image data and show
    Image(user['profile']['image_192'])


channel_dict = {}

for channel in channels['channels']:
    channel_dict[channel['name']] = channel['id']

user_dict = {}

for user in users['members']:
    user_dict[user['name']] = user['id']

print(user_dict)
'''

import sys
non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)

results = [t for t in users['members'] if t['profile']['display_name'] == "chun7642"]
print(results)

me = users['members'][0]
print(str(me['profile']['title']).translate(non_bmp_map))
print(str(me['profile']['display_name']).translate(non_bmp_map))
print(str(me['profile']['status_text']).translate(non_bmp_map))
print(str(me['profile']['status_emoji']).translate(non_bmp_map))


import emoji


def print_emoji(emoji_name):
    print(str(emoji.emojize(emoji_name, use_aliases=True)).translate(non_bmp_map))


print_emoji(':runner:')
print_emoji(':computer:')

'''
import io
import base64
import turtle
import tkinter as tk
from urllib.request import urlopen
root = tk.Tk()
root.title("turtle graphics a website image")
image_url = "https://a.slack-edge.com/c00d19/img/emoji_2017_12_06/sheet_google_64_indexed_256.png"
image_byt = urlopen(image_url).read()
image_b64 = base64.encodestring(image_byt)
photo = tk.PhotoImage(data=image_b64)

# create a white canvas large enough to fit the image+
w = 540
h = 340
cv = tk.Canvas(bg='white', width=w, height=h)
cv.pack()
# this makes the canvas a turtle canvas
# point(0, 0) is in the center now
tu = turtle.RawTurtle(cv)
# put the image on the turtle canvas with
# create_image(xpos, ypos, image, anchor)
xpos = int(w/2 * 0.9)
ypos = int(h/2 * 0.9)
print(xpos, ypos)
cv.create_image(-xpos, -ypos, image=photo, anchor='nw')
'''

def convert_ts(ts):
    return pd.to_datetime(ts, unit='s').date()

matches = slack.search.messages(query='Python').body['messages']['matches']
for match in matches:
    print(f"Time: {convert_ts(match['ts'])}, Text: {match['text']}")
print(str(emoji.emojize("Python is fun :+1:", use_aliases=True)).translate(non_bmp_map))

matches = slack.search.files(query='plot').body['files']['matches']
for match in matches:
    print(f"Time: {convert_ts(match['timestamp'])}, Title: {match['title']}")


print(slack.dnd.team_info().body)

'''
kde plot
'''
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# library & dataset
df = sns.load_dataset('iris')

# Basic 2D density plot
sns.set_style("white")

# Some features are characteristic of 2D: color palette and wether or not color the lowest range
sns.kdeplot(df.sepal_width, df.sepal_length,
            cmap="Blues", shade=True, shade_lowest=True, )
plt.title('Good Ole Iris Data Set')

# Save last figure
plt.savefig('iris_plot.png', dpi=500)


'''
heat map
'''
from string import ascii_letters

d = pd.DataFrame(data=np.random.normal(size=(100, 26)),
                 columns=list(ascii_letters[26:]))

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.savefig('heatmap_ex.png')
