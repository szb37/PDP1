import os

src = os.path.dirname(os.path.abspath(__file__))
codebase = os.path.abspath(os.path.join(src, os.pardir))
data = os.path.abspath(os.path.join(codebase, 'data'))
raw = os.path.abspath(os.path.join(data, 'raw'))
processed = os.path.abspath(os.path.join(data, 'processed'))
tmp = os.path.abspath(os.path.join(codebase, 'tmp'))

exports = os.path.abspath(os.path.join(codebase, 'exports'))
ind_timeevols = os.path.abspath(os.path.join(exports, 'timeevol ind figs'))
agg_timeevols = os.path.abspath(os.path.join(exports, 'timeevol agg figs'))
histograms = os.path.abspath(os.path.join(exports, 'histogram figs'))
vitals = os.path.abspath(os.path.join(exports, 'vitals figs'))
fivedasc = os.path.abspath(os.path.join(exports, '5dasc'))
cytokine = os.path.abspath(os.path.join(exports, 'cytokine corr matrix'))
