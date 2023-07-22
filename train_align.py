'''
code for entity alignment task
'''
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from metrics import *
from models import AutoRGCN_Align
import logging
import os

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'wordnet', 'Dataset name: zh_en, ja_en, fr_en')
flags.DEFINE_string('mode', 'None', 'KE method for GCN: TransE, TransH, TransD, DistMult, RotatE, QuatE')
flags.DEFINE_string('optim', 'Adam', 'Optimizer: GD, Adam')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs for training.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('gamma', 3.0, 'Hyper-parameter for margin based loss.')
flags.DEFINE_integer('num_negs', 5, 'Number of negative samples for each positive seed.')
flags.DEFINE_float('alpha', 0.5, 'Weight of entity conv update.')
flags.DEFINE_float('beta', 0.5, 'Weight of relation conv update.')
flags.DEFINE_integer('layer', 0, 'number of hidden layers')
flags.DEFINE_integer('dim', 200, 'hidden Dimension')
flags.DEFINE_integer('seed', 3, 'Proportion of seeds, 3 means 30%')
flags.DEFINE_boolean('rel_align', True, 'If true, use relation alignment information.')
flags.DEFINE_boolean('rel_update', False, 'If true, use graph conv for rel update.')
flags.DEFINE_integer('randomseed', 12306, 'seed for randomness')
flags.DEFINE_boolean('valid', False, 'If true, split validation data.')
flags.DEFINE_boolean('save', False, 'If true, save the print')
flags.DEFINE_string('metric', "cityblock", 'metric for testing')
flags.DEFINE_string('loss_mode', "L1", 'mode for loss calculation')
flags.DEFINE_string('embed', "random", 'init embedding for entities') # random, text

np.random.seed(FLAGS.randomseed)
random.seed(FLAGS.randomseed)
tf.set_random_seed(FLAGS.randomseed)

if FLAGS.save:
    nsave = "log/{}/{}".format(FLAGS.dataset, FLAGS.mode)
else:
    print("not saving file")
    nsave = "log/trash"
create_exp_dir(nsave)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p', filemode="w")
save_fname = 'alpha{}-beta{}-layer{}-sdim{}-lr{}-seed{}'.format(
               FLAGS.alpha, FLAGS.beta, FLAGS.layer, FLAGS.dim,
               FLAGS.learning_rate, FLAGS.randomseed)

save_fname = "auto-" + save_fname
if not FLAGS.valid:
    save_fname = "test-" + save_fname
fh = logging.FileHandler(os.path.join(nsave, save_fname + ".txt"), "w")
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.getLogger().setLevel(logging.INFO)

# Load data
adj, num_ent, train, test, valid, y = load_data_class(FLAGS)
# y = np.ones((len(train),1))
negative_adj = get_negatives(adj, test)
negative_index = [i-len(negative_adj) for i in range(len(negative_adj))]
# negative_index = negative_index[:200]
train = train + negative_index
train = [train, y, adj[-1], negative_adj]
rel_num = np.max(adj[2][:, 1]) + 1
print("Relation num: ", rel_num)

# process graph to fit into later computation
support = [preprocess_adj(adj)]
num_supports = 1
model_func = AutoRGCN_Align
num_negs = FLAGS.num_negs
print("Entity num: ", num_ent)

# Define placeholders
placeholders = {
    'features': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder_with_default(0, shape=())
}
# graph structure data
placeholders['support'] = [[tf.placeholder(tf.float32, shape=[None, 1]),
                    tf.placeholder(tf.float32, shape=[None, 1]), \
                    tf.placeholder(tf.int32)] for _ in range(num_supports)]

# Create model
input_dim = [num_ent, rel_num]
hidden_dim = [FLAGS.dim, FLAGS.dim]
output_dim = [FLAGS.dim, FLAGS.dim]
if FLAGS.mode == "TransH":
    hidden_dim[1] *= 2
    output_dim[1] *= 2
elif FLAGS.mode == "TransD":
    hidden_dim[0] *= 2
    hidden_dim[1] *= 2
    output_dim[0] *= 2
    output_dim[1] *= 2
# names_neg = [["left", "neg_right", "neg_left", "right"]]
names_neg = [["true", "neg"]]
model = model_func(placeholders, input_dim, hidden_dim, output_dim, dataset=FLAGS.dataset,
                    train_labels=train, REL=None, mode=FLAGS.mode, embed=FLAGS.embed, alpha=FLAGS.alpha,
                    beta=FLAGS.beta, layer_num=FLAGS.layer, sparse_inputs=False, featureless=True,
                    logging=True, rel_update=FLAGS.rel_update, task="link", names_neg=names_neg)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# generate positive examples
# num_labels = len(train)
# labels = np.ones((num_labels, 1))


# Train model
for epoch in range(FLAGS.epochs):
    # generate negtive examples
    # if epoch % 10 == 0:
    #     neg = np.random.choice(num_ent, num_labels * num_negs)
    feed_dict = construct_feed_dict(1.0, support, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # for i,labels in enumerate([labels,negatives]):
    #     feed_dict.update({names_neg[0][i]+":0": labels})
    # Training step
    outputs = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)

    # Print results
    if epoch % 10 == 0:
        logging.info("Epoch: {} train_loss= {:.5f}".format(epoch+1, outputs[1]))

    if epoch % 50 == 0 and valid is not None:
        output_embeddings = sess.run(model.outputs, feed_dict=feed_dict)
        get_link(output_embeddings[0], valid, negative_adj[:200], adj[-1], logging)


    if epoch % 50 == 0 and epoch > 0 and valid is None:
        output_embeddings = sess.run(model.outputs, feed_dict=feed_dict)
        get_link(output_embeddings[0], test, negative_adj[:200], adj[-1], logging)

print("Optimization Finished!")

if valid is not None:
    exit()

# test
# feed_dict = construct_feed_dict(1.0, support, placeholders)
# output_embeddings = sess.run(model.outputs, feed_dict=feed_dict)
# get_linl(output_embeddings[0], test, logging, FLAGS.metric)
