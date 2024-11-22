# Where the datasets are
DATA_DIR = 'data'
DATASETS_JSON = DATA_DIR + '/datasets.json'

KERAS_TUNER = True
KERAS_TUNER_TRIALS = 2

ENCODING_TYPE = 'Try Both' # need to pass 'One Hot' or 'Linear' or 'Try Both'

# Enable data augmentation/scaling, etc
DATA_AUGMENTATION = True
LOG_DRAIN_CURRENT = True

# We use a simple fully connected network (MLP) 
# 4 layers because deeper NNs can capture more complex patterns
# Gradually decrease the neuron size to better capture patterns while avoiding overfitting
NEURONS_PER_LAYER = [3300, 200, 4600, 1700]
TRAIN_DROPOUT_RATE = 0.4

# Training hyper-parameters

# Learning Rate gives the step size that the optimizer takes while learning, 
# smaller step size means slower convergence but more accuracy
LR_INITIAL = 0.0014558 #0.00020597

# Learning rate decay helps the model become refined as it gets closer to a minimum
# The learning rate decay steps desides how many steps the learning rate will decay after
LR_DECAY_STEPS = 100  # 100 best for log phig1 cadence data

# LR_INITIAL * LR_DECAY_RATE after each number of LR_DECAY_STEPS
LR_DECAY_RATE = 0.99

# Staircase or continuous?
LR_STAIRCASE = False

TRAIN_EARLY_STOPPING_PATIENCE = 50
TRAIN_BATCH_SIZE = 32 # 32 default
#TRAIN_VALIDATION_SPLIT = 0.2

TRAIN_LOSS = 'mean_squared_error'
#TRAIN_LOSS = 'mean_squared_logarithmic_error'



