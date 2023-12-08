import os

src = os.path.dirname(os.path.abspath(__file__))
codebase = os.path.abspath(os.path.join(src, os.pardir))
data = os.path.abspath(os.path.join(codebase, 'data'))
raw_data = os.path.abspath(os.path.join(data, 'raw'))
processed = os.path.abspath(os.path.join(data, 'processed'))

exports = os.path.abspath(os.path.join(codebase, 'exports'))
timeevols_dir = os.path.abspath(os.path.join(exports, 'time evolution figs'))
histograms_dir = os.path.abspath(os.path.join(exports, 'histogram figs'))
vitals_dir = os.path.abspath(os.path.join(exports, 'vitals figs'))
