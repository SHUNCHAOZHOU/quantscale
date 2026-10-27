"""Export the verified reconstruction of the selected policies' validation paths."""
from .plot_v3_cumulative_returns import export

if __name__=='__main__':
    export(validation=True)
