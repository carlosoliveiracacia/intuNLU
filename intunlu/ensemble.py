from intunlu.generate_ensemble import *

model = None
document = None
device = None
max_input_length = None
pred = generate([model], document, device, max_input_length)
