import os

src = os.path.dirname(os.path.abspath(__file__))
codebase = os.path.abspath(os.path.join(src, os.pardir))
data = os.path.abspath(os.path.join(codebase, 'data'))
exports = os.path.abspath(os.path.join(codebase, 'exports'))
