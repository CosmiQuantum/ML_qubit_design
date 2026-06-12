from pathlib import Path

# transmon cross cap matrix

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
ARTIFACT_DIR_CANDIDATES = [
    EXPERIMENT_DIR,
    REPO_ROOT / 'experiments' / 'model_predict_qubit-TransmonCross-Hamiltonian_params',
    REPO_ROOT / 'model_predict_qubit_TransmonCross_Hamiltonian_params',
]
ARTIFACT_DIR = next(
    (
        path for path in ARTIFACT_DIR_CANDIDATES
        if any((path / child).exists() for child in ('model', 'data', 'scalers'))
    ),
    EXPERIMENT_DIR,
)

DATA_DIR = str(ARTIFACT_DIR / 'data')
SCALERS_DIR = str(ARTIFACT_DIR / 'scalers')
MODEL_DIR = str(ARTIFACT_DIR / 'model')
KERAS_DIR = str(ARTIFACT_DIR / 'keras')
KT_DIR = str(ARTIFACT_DIR / 'kt_dir2')
METADATA_DIR = str(EXPERIMENT_DIR / 'metadata')
RESULTS_DIR = str(EXPERIMENT_DIR / 'results')
PLOTS_DIR = str(EXPERIMENT_DIR / 'plots')
DATASETS_JSON = str(Path(DATA_DIR) / 'datasets.json')

SWEEP_PARAM_NUM = False
SWEEP_DATA_AMOUNT = False
VISUALIZE_GRADIENTS = False

KERAS_TUNER = True
KERAS_TUNER_TRIALS = 2000
ENCODING_TYPE = 'one hot' # need to pass 'one hot' or 'linear' or 'Try Both'

# Enable data augmentation/scaling, etc
DATA_AUGMENTATION = True

# We use a simple fully connected network (MLP)
# Architecture sweep winner from ml_21:
# depth = 3 hidden layers, width = 32 neurons per hidden layer
NEURONS_PER_LAYER = [64]#[32, 32, 32]
TRAIN_DROPOUT_RATE = 0 #0.05

# Training hyperparameters

# Learning Rate gives the step size that the optimizer takes while learning,
# smaller step size means slower convergence but more accuracy
# learning rate is=LR_INITIAL×(LR_DECAY_RATE)^(t/LR_DECAY_STEPS)
LR_INITIAL = 0.001

# Learning rate decay helps the model become refined as it gets closer to a minimum
# The learning rate decay steps desides how many steps the learning rate will decay after

# LR_DECAY_STEPS = 35 # 100 best for log phig1 cadence data

# LR_INITIAL * LR_DECAY_RATE after each number of LR_DECAY_STEPS
LR_DECAY_RATE = 0.99

# Staircase or continuous?
LR_STAIRCASE = False

EPOCHS = 400

TRAIN_EARLY_STOPPING_PATIENCE = 60
TRAIN_BATCH_SIZE = 128 # 32 default
#TRAIN_VALIDATION_SPLIT = 0.2

#TRAIN_LOSS = 'mean_squared_error'
TRAIN_LOSS = 'mae' # mean absolute error
#TRAIN_LOSS = 'mean_squared_logarithmic_error'
