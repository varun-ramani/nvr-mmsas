import seaborn as sns
import pandas as pd
from IPython import display
from PIL import Image

def filter_on(data, combo=None, lr=None, ls=None, lp=None, ltv=None):
    # start with vacuous condition
    condition = pd.Series([True] * len(data))
    for key, val in [
        ('combo', combo),
        ('lr', lr),
        ('ls', ls),
        ('lp', lp),
        ('ltv', ltv)
    ]:  
        if val is not None:
            condition &= (data[key] == val)

    return data.loc[condition]

def visualize_combos(data_src, filtered_data, display_fn=display):
    for index, data in filtered_data.iterrows():
        print(data)
        display(Image.open(data_src / data['combo'] / 'signal_result.png'))
        display(Image.open(data_src / data['combo'] / 'loss_components_over_epochs.png'))