from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from modules.models import PRNet_Model
from modules.lr_scheduler import MultiStepLR
from modules.losses import WeightedMSE
from modules.utils import load_yaml, ProgressBar, set_memory_growth
from modules.dataset import load_dataset, take

flags.DEFINE_string('cfg_path', './configs/prnet.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')


def main(_):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    set_memory_growth()

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)

    cfg = load_yaml(FLAGS.cfg_path)
    uv_weight_mask = cv2.imread(cfg['uv_weight_mask']) / 255.

    # define network
    generator = PRNet_Model(cfg['input_size'], cfg['ch_size'])
    generator.summary(line_length=80)

    # load dataset
    train_dataset = load_dataset(cfg,
                                 shuffle=True, num_workers=cfg['num_workers'])

    # define optimizer
    learning_rate_G = MultiStepLR(cfg['lr_G'], cfg['lr_steps'], cfg['lr_rate'])
    optimizer_G = tf.keras.optimizers.Adam(learning_rate=learning_rate_G,
                                           beta_1=cfg['adam_beta1_G'],
                                           beta_2=cfg['adam_beta2_G'])

    # define losses function
    loss_fn = WeightedMSE(uv_weight_mask)

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     optimizer_G=optimizer_G,
                                     model=generator)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print('[*] load ckpt from {} at step {}.'.format(
            manager.latest_checkpoint, checkpoint.step.numpy()))
    else:
        if cfg['pretrain_name'] is not None:
            pretrain_dir = './checkpoints/' + cfg['pretrain_name']
            if tf.train.latest_checkpoint(pretrain_dir):
                checkpoint.restore(tf.train.latest_checkpoint(pretrain_dir))
                checkpoint.step.assign(0)
                print("[*] training from pretrain model {}.".format(
                    pretrain_dir))
            else:
                print("[*] cannot find pretrain model {}.".format(
                    pretrain_dir))
        else:
            print("[*] training from scratch.")

    # define training step function
    @tf.function
    def train_step(img, pos):
        with tf.GradientTape() as tape_G:
            pre = generator(img, training=True)

            losses_G = {}
            losses_G['pixel'] = loss_fn(pos, pre)
            total_loss_G = tf.add_n([l for l in losses_G.values()])

        grads_G = tape_G.gradient(
            total_loss_G, generator.trainable_variables)
        optimizer_G.apply_gradients(
            zip(grads_G, generator.trainable_variables))

        return total_loss_G, losses_G

    # training loop
    summary_writer = tf.summary.create_file_writer(
        './logs/' + cfg['sub_name'])
    niter = int(cfg['train_dataset']['num_samples'] * cfg['epoch'] /
                cfg['batch_size'])
    prog_bar = ProgressBar(niter, checkpoint.step.numpy())
    remain_steps = max(niter - checkpoint.step.numpy(), 0)

    for sample in take(remain_steps, train_dataset):
        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()
        img, pos = sample['Image'], sample['Posmap']

        total_loss_G, losses_G = train_step(img, pos)

        prog_bar.update_gan(total_loss_G.numpy(),
                            optimizer_G.lr(steps).numpy())

        if steps % cfg['log_steps'] == 0:
            with summary_writer.as_default():
                tf.summary.scalar(
                    'loss_G/total_loss', total_loss_G, step=steps)
                for k, l in losses_G.items():
                    tf.summary.scalar('loss_G/{}'.format(k), l, step=steps)

                tf.summary.scalar(
                    'learning_rate_G', optimizer_G.lr(steps), step=steps)

        if steps % cfg['save_steps'] == 0:
            manager.save()
            print('\n[*] save ckpt file at {}'.format(
                manager.latest_checkpoint))

    print("\n[*] training done!")


if __name__ == '__main__':
    app.run(main)
