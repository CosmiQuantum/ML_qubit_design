# Where the datasets are
DATA_DIR = 'data'
DATASETS_JSON = DATA_DIR + '/datasets.json'

SWEEP_PARAM_NUM = False
SWEEP_DATA_AMOUNT = False
VISUALIZE_GRADIENTS = True

KERAS_TUNER = True
KERAS_TUNER_TRIALS = 854
KERAS_DIR = 'keras'

ENCODING_TYPE = 'one hot' # need to pass 'one hot' or 'linear' or 'Try Both'

# Enable data augmentation/scaling, etc
DATA_AUGMENTATION = True

# We use a simple fully connected network (MLP) 
# 4 layers because deeper NNs can capture more complex patterns
# Gradually decrease the neuron size to better capture patterns while avoiding overfitting
NEURONS_PER_LAYER = [1024]
TRAIN_DROPOUT_RATE = 0.05

# Training hyper-parameters

# Learning Rate gives the step size that the optimizer takes while learning, 
# smaller step size means slower convergence but more accuracy
# learning rate is=LR_INITIAL×(LR_DECAY_RATE)^(t/LR_DECAY_STEPS)
LR_INITIAL = 0.000982

# Learning rate decay helps the model become refined as it gets closer to a minimum
# The learning rate decay steps desides how many steps the learning rate will decay after
LR_DECAY_STEPS = 35  # 100 best for log phig1 cadence data

# LR_INITIAL * LR_DECAY_RATE after each number of LR_DECAY_STEPS
LR_DECAY_RATE = 0.99

# Staircase or continuous?
LR_STAIRCASE = False

TRAIN_EARLY_STOPPING_PATIENCE = 400
TRAIN_BATCH_SIZE = 128 # 32 default
#TRAIN_VALIDATION_SPLIT = 0.2

#TRAIN_LOSS = 'mean_squared_error'
TRAIN_LOSS = 'mae' # mean absolute error
#TRAIN_LOSS = 'mean_squared_logarithmic_error'



