from pathlib import Path

# resonator/cavity claw route meander

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
ARTIFACT_DIR = REPO_ROOT / 'model_predict_cavity_claw_RouteMeander_eigenmode'
if not ARTIFACT_DIR.exists():
    ARTIFACT_DIR = EXPERIMENT_DIR

DATA_DIR = str(ARTIFACT_DIR / 'data')
SCALERS_DIR = str(ARTIFACT_DIR / 'scalers')
MODEL_DIR = str(ARTIFACT_DIR / 'model')
KERAS_DIR = str(ARTIFACT_DIR / 'keras')
KT_DIR = str(EXPERIMENT_DIR / 'kt_dir')
METADATA_DIR = str(EXPERIMENT_DIR / 'metadata')
RESULTS_DIR = str(EXPERIMENT_DIR / 'results')
PLOTS_DIR = str(EXPERIMENT_DIR / 'plots')
DATASETS_JSON = str(Path(DATA_DIR) / 'datasets.json')

SWEEP_PARAM_NUM = False
SWEEP_DATA_AMOUNT = False
VISUALIZE_GRADIENTS = False

KERAS_TUNER = True
KERAS_TUNER_TRIALS = 222
ENCODING_TYPE = 'one hot' # need to pass 'one hot' or 'linear' or 'Try Both'

# Enable data augmentation/scaling, etc
DATA_AUGMENTATION = True

# We use a simple fully connected network (MLP)
# 4 layers because deeper NNs can capture more complex patterns
# Gradually decrease the neuron size to better capture patterns while avoiding overfitting
NEURONS_PER_LAYER = [430,550,580,270,550]#[570,510,250,440,480]
TRAIN_DROPOUT_RATE = 0#0.05

# Training hyperparameters

# Learning Rate gives the step size that the optimizer takes while learning,
# smaller step size means slower convergence but more accuracy
# learning rate is=LR_INITIAL×(LR_DECAY_RATE)^(t/LR_DECAY_STEPS)
LR_INITIAL = 0.0049807

# Learning rate decay helps the model become refined as it gets closer to a minimum
# The learning rate decay steps desides how many steps the learning rate will decay after
# LR_DECAY_STEPS = 35 # 100 best for log phig1 cadence data

# LR_INITIAL * LR_DECAY_RATE after each number of LR_DECAY_STEPS
LR_DECAY_RATE = 0.99

# Staircase or continuous?
LR_STAIRCASE = False

TRAIN_EARLY_STOPPING_PATIENCE = 60
TRAIN_BATCH_SIZE = 128 # 32 default
#TRAIN_VALIDATION_SPLIT = 0.2

#TRAIN_LOSS = 'mean_squared_error'
TRAIN_LOSS = 'mae' # mean absolute error
#TRAIN_LOSS = 'mean_squared_logarithmic_error'

EPOCHS = 400
