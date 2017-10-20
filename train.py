import os
import cntk
import argparse
import numpy as np
from cntk.train import Trainer
from cntk.learners import IGNORE
from cntk import learning_parameter_schedule, momentum_schedule
from cntk.io import StreamDef, StreamDefs, MinibatchSource, CBFDeserializer
from cntk.layers import Convolution2D, Activation, MaxPooling, Dense, Dropout, default_options, Sequential, For, LSTM, Recurrence

# Model dimensions
frame_height = 147
frame_width = 133
num_channels = 1
num_classes = 2746
hidden_dim = 4096
model_name = 'asllvd.model'

# Dataset partition sizes (in sequences)
test_size = 2857
train_size = 6875


def cbf_reader(path, is_training):
    """
    Returns a MinibatchSource for data at the given path
    :param path: Path to a CBF file
    :param is_training: Set to true if reader is for training set, else false
    """
    return MinibatchSource(CBFDeserializer(path, StreamDefs(
        label=StreamDef(field='label', shape=num_classes, is_sparse=True),
        pixels=StreamDef(field='pixels', shape=frame_height * frame_width * num_channels, is_sparse=False)
    )), randomize=is_training, max_sweeps=cntk.io.INFINITELY_REPEAT if is_training else 1)


def conv_model():
    with default_options(activation=None, pad=True, bias=True):
        model = Sequential([
            # Separate Convolution and ReLU for feature extraction
            For(range(2), lambda i: [
                Convolution2D((3, 3), 32, name='conv1_{}'.format(i)),
                Activation(activation=cntk.relu, name='relu1_{}'.format(i))
            ]),
            MaxPooling((2, 2), (2, 2), name='pool1'),

            For(range(2), lambda i: [
                Convolution2D((3, 3), 64, name='conv2_{}'.format(i)),
                Activation(activation=cntk.relu, name='relu2_{}'.format(i))
            ]),
            MaxPooling((2, 2), (2, 2), name='pool2'),

            For(range(4), lambda i: [
                Convolution2D((3, 3), 128, name='conv3_{}'.format(i)),
                Activation(activation=cntk.relu, name='relu3_{}'.format(i))
            ]),
            MaxPooling((2, 2), (2, 2), name='pool3'),

            For(range(4), lambda i: [
                Convolution2D((3, 3), 64, name='conv4_{}'.format(i)),
                Activation(activation=cntk.relu, name='relu4_{}'.format(i))
            ]),
            MaxPooling((2, 2), (2, 2), name='pool4'),

            For(range(4), lambda i: [
                Convolution2D((3, 3), 32, name='conv5_{}'.format(i)),
                Activation(activation=cntk.relu, name='relu5_{}'.format(i))
            ]),
            MaxPooling((2, 2), (2, 2), name='pool5'),

            Dense(8192, name='fc6'),
            Activation(activation=cntk.relu, name='relu6'),
            Dropout(0.2, name='drop6'),
        ])
    return model


def create_model():
    """
    Creates the network to be trained
    :return dictionary of input variable, target variable, model, loss and metric functions.
    """
    # Create the input and target variables
    input_var = cntk.sequence.input_variable((1, frame_height, frame_width), name='input_var')
    target_var = cntk.input_variable((num_classes,), is_sparse=True, name='target_var')

    conv = conv_model()
    with cntk.default_options(activation=cntk.relu, go_backwards=False):
        rec_block = LSTM(hidden_dim, name='conv_lstm')

        @cntk.Function
        def cnn_lstm(dh, dc, x):
            dx = conv(x)
            return rec_block(dh, dc, dx)

        model = Sequential([
            Recurrence(cnn_lstm, name='rec7'),
            cntk.sequence.last,
            Dropout(0.3, name='drop8'),
            Dense(num_classes, activation=None, name='fc9')
        ])(input_var)

    return {
        'input': input_var,
        'target': target_var,
        'model': model,
        'loss': cntk.cross_entropy_with_softmax(model, target_var),
        'metric': cntk.classification_error(model, target_var)
    }


def train(params, network, trainer, writers):
    # Create the dataset readers
    train_reader = cbf_reader(os.path.join(params['input_folder'], 'train.cbf'), is_training=True)
    cv_reader = cbf_reader(os.path.join(params['input_folder'], 'test.cbf'), is_training=True)

    input_map = {
        network['input']: train_reader.streams.pixels,
        network['target']: train_reader.streams.label
    }

    # Restore model and trainer from checkpoint if requested
    chkpt_path = os.path.join(params['output_folder'], model_name)
    if params['restore']:
        trainer.restore_from_checkpoint(chkpt_path)

    tb_writer = writers['tb']
    cntk.logging.log_number_of_parameters(network['model'])

    for epoch in range(params['max_epochs']):
        minibatch = 0
        sequences = 0
        while sequences < train_size:
            mb = train_reader.next_minibatch(params['minibatch_size'], input_map)
            trainer.train_minibatch(mb)

            minibatch += 1
            sequences += mb[network['input']].num_sequences

            if minibatch % 500 == 0:
                tb_writer.flush()

        # Write per-epoch statistics
        trainer.summarize_training_progress()
        tb_writer.flush()

        # Checkpoint after each epoch
        trainer.save_checkpoint(chkpt_path)

        # Perform cross validation on subset of test set
        metrics = []
        sequences = 0

        while sequences < params['cv_seqs']:
            mb = cv_reader.next_minibatch(params['minibatch_size'], input_map)
            metrics.append(trainer.test_minibatch(mb))
            sequences += mb[network['input']].num_sequences

        ave = np.average(metrics)
        tb_writer.write_value('cv/ave_metric', ave, epoch + 1)
        tb_writer.flush()

        writers['pp'].write('Cross Validation', '{}'.format(ave))


def test(params, network, writers):
    model = network['model']
    test_reader = cbf_reader(os.path.join(params['input_folder'], 'test.cbf'), is_training=False)
    input_map = {
        network['input']: test_reader.streams.pixels,
        network['target']: test_reader.streams.label
    }

    wrong = 0
    sequences = 0
    while True:
        mb = test_reader.next_minibatch(params['minibatch_size'], input_map)
        if len(mb) == 0:
            break

        prediction = model.eval(mb[network['input']])
        targets = mb[network['target']].asarray()

        seqs = mb[network['input']].num_sequences
        for i in range(seqs):
            if np.argmax(prediction[i]) != np.argmax(targets[i]):
                wrong += 1

        sequences += seqs

    error = wrong / sequences * 100.0
    writers['tb'].write_value('test/error', error, 0)
    writers['pp'].write('Test', 'Final error: {}'.format(error))


def main(params):
    # Create output and log directories if they don't exist
    if not os.path.isdir(params['output_folder']):
        os.makedirs(params['output_folder'])

    if not os.path.isdir(params['log_folder']):
        os.makedirs(params['log_folder'])

    # Create the network
    network = create_model()

    # Create learner
    l2_reg_weight = 0.0005
    lr_schedule = learning_parameter_schedule(params['sample_lr'], minibatch_size=IGNORE)
    mm_schedule = momentum_schedule(0.99)
    learner = cntk.momentum_sgd(network['model'].parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)

    # Use TensorBoard for visual logging
    pp_writer = cntk.logging.ProgressPrinter(freq=50, tag='Training', num_epochs=params['max_epochs'])
    tb_writer = cntk.logging.TensorBoardProgressWriter(freq=50, log_dir=params['log_folder'], model=network['model'])
    writers = {
        'pp': pp_writer,
        'tb': tb_writer
    }

    # Create trainer and training session
    trainer = Trainer(network['model'], (network['loss'], network['metric']), [learner], [pp_writer, tb_writer])

    try:
        train(params, network, trainer, writers)
        test(params, network, writers)
    finally:
        path = os.path.join(params['output_folder'], 'final_model.dnn')
        network['model'].save(path)
        print('Saved final model to', path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-if', '--input_folder', help='Directory where dataset is located', required=False, default='dataset')
    parser.add_argument('-of', '--output_folder', help='Directory for models and checkpoints', required=False, default='models')
    parser.add_argument('-lf', '--log_folder', help='Directory for log files', required=False, default='logs')
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False, default=100)
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size in samples', type=int, required=False, default=128)
    parser.add_argument('-e', '--epoch_size', help='Epoch size in sequences', type=int, required=False, default=6875)
    parser.add_argument('-r', '--restore', help='Indicates whether to resume from previous checkpoint', action='store_true')
    parser.add_argument('-lr', '--lr_per_sample', help='Per sample learning rate', type=float, required=False, default=0.0008)
    parser.add_argument('-c', '--cv_seqs', help='The number of sequences to use for cross validation', required=False, default=600)

    args = parser.parse_args()
    main({
        'input_folder': args.input_folder,
        'output_folder': args.output_folder,
        'log_folder': args.log_folder,
        'max_epochs': args.num_epochs,
        'minibatch_size': args.minibatch_size,
        'epoch_size': args.epoch_size,
        'restore': args.restore,
        'sample_lr': args.lr_per_sample,
        'cv_seqs': args.cv_seqs
    })
