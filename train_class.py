'''
code for entity classification task
'''
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from metrics import *
from models import AutoRGCN_Align
import random
import logging
import os
import tensorflow_privacy

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy, rdp_accountant
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent

# optimizer = tensorflow_privacy.DPAdamGaussianOptimizer(
#                 l2_norm_clip=1.5,
#                 noise_multiplier=1.3,
#                 num_microbatches=250,
#                 learning_rate=.25)
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'fb15k', 'Dataset: am, wordnet.')
flags.DEFINE_string('mode', 'TransE', 'KE method for GCN: TransE, TransH, TransD, DistMult, RotatE, QuatE')
# flags.DEFINE_string('optim', 'Adam', 'Optimizer: GD, Adam')
flags.DEFINE_string('optim', 'DPSGD', 'Optimizer: DPSGD, GD, Adam')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('epochss', 200, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('gamma', 3.0, 'Hyper-parameter for margin based loss.')
flags.DEFINE_integer('num_negs', 5, 'Number of negative samples for each positive seed.')
flags.DEFINE_float('alpha', 0.5, 'Weight of entity conv update.')
flags.DEFINE_float('beta', 0.5, 'Weight of relation conv update.')
flags.DEFINE_integer('layer', 2, 'number of hidden layers')
flags.DEFINE_integer('dim', 32, 'hidden Dimension')
flags.DEFINE_integer('randomseed', 12306, 'seed for randomness')
flags.DEFINE_boolean('rel_update', False, 'If true, use graph conv for rel update.')
flags.DEFINE_boolean('valid', False, 'If true, split validation data.')
flags.DEFINE_boolean('save', False, 'If true, save the print')
flags.DEFINE_string('metric', "cityblock", 'metric for testing')
flags.DEFINE_string('loss_mode', "L1", 'mode for loss calculation')
flags.DEFINE_string('embed', "random", 'init embedding for entities')

# flags.DEFINE_float('l2_norm_clip', 1.0, 'Initial learning rate.', allow_override= True)
# flags.DEFINE_float('noise_multiplier_flag', 0.1, 'Initial learning rate.', allow_override= True)

np.random.seed(FLAGS.randomseed)
random.seed(FLAGS.randomseed)
tf.set_random_seed(FLAGS.randomseed)

eps, order = compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=1344,
                                              batch_size=1344,
                                              noise_multiplier=2.0,
                                              epochs=160,
                                              delta=1e-5)

print (eps)
orders = [1 + x / 10. for x in range(1, 1000)] + list(range(12, 64))
rdp = compute_rdp(q=0.25,
                  noise_multiplier=2.0,
                  steps=160,
                  orders=orders)

epsilon = get_privacy_spent(orders, rdp, target_delta=1e-5)[0]
print (epsilon)

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
logging.info("Train len: {}".format(len(train)))
logging.info("Test len: {}".format(len(test)))
logging.info("Valid len: {}".format(len(valid)))
print(y.shape)
train = [train, y]
rel_num = np.max(adj[2][:, 1]) + 1
print("Relation num: ", rel_num)

# process graph to fit into later computation
support = [preprocess_adj(adj)]
num_supports = 1
model_func = AutoRGCN_Align
num_negs = FLAGS.num_negs
class_num = y.shape[1]
print("Entity num: ", num_ent)
print("Class num: ", class_num)

logging.info("entity num: {}".format(num_ent))
logging.info("class num: {}".format(class_num))


if FLAGS.dataset == "fb15k":
    task = "label"
    get_eval = get_label
elif FLAGS.dataset == "wn":
    task = "link"
    get_eavl = get_link
else:
    task = "class"
    get_eval = get_class

# Define placeholders
placeholders = {
    'features': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder_with_default(0, shape=())
}
placeholders['support'] = [[tf.placeholder(tf.float32, shape=[None, 1]),
                    tf.placeholder(tf.float32, shape=[None, 1]), \
                    tf.placeholder(tf.int32)] for _ in range(num_supports)]

# Create model
input_dim = [num_ent, rel_num]
hidden_dim = [FLAGS.dim, FLAGS.dim]
output_dim = [class_num, FLAGS.dim]
if FLAGS.mode == "TransH":
    hidden_dim[1] *= 2
elif FLAGS.mode == "TransD":
    hidden_dim[0] *= 2
    hidden_dim[1] *= 2
model = model_func(placeholders, input_dim, hidden_dim, output_dim, dataset=FLAGS.dataset,
                    train_labels=train, mode=FLAGS.mode, embed=FLAGS.embed, alpha=FLAGS.alpha,
                    beta=FLAGS.beta, layer_num=FLAGS.layer, sparse_inputs=False, featureless=True,
                    logging=True, rel_update=FLAGS.rel_update, task=task)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
acc_best = 0.
test_acc = 0.
    
    

# Train model
for epoch in range(FLAGS.epochss):

    # Construct feed dictionary
    feed_dict = construct_feed_dict(1.0, support, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Training step
    outputs = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)
    # Print results
    if epoch % 10 == 0:
        logging.info(f"Epoch: {epoch+1} train_loss= {outputs[1]}")

    if epoch % 10 == 0 and valid is not None:
        # model.evaluate()
        output_embeddings = sess.run(model.outputs, feed_dict=feed_dict)
        train_acc, _ = get_eval(output_embeddings[0], train[0], y, logging)
        logging.info("Train Accuracy: %.3f" % (train_acc * 100))
        acc, _ = get_eval(output_embeddings[0], valid, y, logging)
        logging.info("Valid Accuracy: %.3f" % (acc * 100))
        if acc > acc_best:
            acc_best = acc
            test_acc, result = get_eval(output_embeddings[0], test, y, logging)
        logging.info("Test Accuracy: %.3f" % (test_acc * 100))


    if epoch % 10 == 0 and epoch > 0 and valid is None:
        # model.evaluate()
        output_embeddings = sess.run(model.outputs, feed_dict=feed_dict)
        train_acc, _ = get_eval(output_embeddings[0], train[0], y, logging)
        logging.info("Train Accuracy: %.3f" % (train_acc * 100))
        acc, temp = get_eval(output_embeddings[0], test, y, logging)
        logging.info("Test Accuracy: %.3f" % (acc * 100))
        if acc > acc_best:
            acc_best = acc
            result = temp

logging.info("Optimization Finished! Best Valid Acc: {} Test: {}".format(
                round(acc * 100,2), " ".join([str(round(i*100,2)) for i in result])))


# eps, order = compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=len(train[0]),
#                                               batch_size=len(train[0]),
#                                               noise_multiplier=100.0,
#                                               epochs=FLAGS.epochss,
#                                               delta=1e-4)

# orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
# rdp = compute_rdp(q=1.0,
#                   noise_multiplier=FLAGS.noise_multiplier,
#                   steps=FLAGS.epochss,
#                   orders=orders)

# logging.info("Final RDP-eps: {}".format(eps))  
