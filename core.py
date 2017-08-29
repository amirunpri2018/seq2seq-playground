import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from models import EncoderRNN, LuongAttnDecoderRNN

USE_CUDA = 1


def _train(input_batch, target_batch, encoder, decoder, encoder_optimizer, decoder_optimizer,
           criterion, clip=None, teacher_forcing_ratio=1):
    input_variable = Variable(torch.from_numpy(
        input_batch).unsqueeze(2).float())
    if USE_CUDA:
        input_variable = input_variable.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each time step

    encoder.train()
    decoder.train()

    # Run sequences through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable)

    # Prepare input and output variables
    decoder_input = Variable(torch.zeros(
        1, input_batch.shape[1], 1))
    # decoder_input = input_variable[-1, :, :].unsqueeze(0)
    # Use last (forward) hidden state from encoder
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    all_decoder_outputs = Variable(torch.zeros(
        target_batch.shape[0], target_batch.shape[1]))
    target_variable = Variable(torch.from_numpy(
        target_batch).unsqueeze(2).float())

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()
        target_variable = target_variable.cuda()

    # Run through decoder one time step at a time
    for t in range(target_batch.shape[0]):
        decoder_output, decoder_hidden, _ = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        all_decoder_outputs[t] = decoder_output[:, 0]
        if np.random.random() < teacher_forcing_ratio:
            decoder_input = target_variable[t, :, :].unsqueeze(0)
        else:
            # Next input is current output
            decoder_input = decoder_output.unsqueeze(0)

    # Loss calculation and backpropagation
    loss = criterion(
        all_decoder_outputs,
        target_variable
    )
    loss.backward()

    if clip is not None:
        # Clip gradient norms
        torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0], all_decoder_outputs


def train(x, y, optimizer=optim.Adam,  criterion=nn.MSELoss(),
          n_steps=100, attn_model="general",
          hidden_size=128, n_layers=1, dropout=0, batch_size=50,
          elr=0.001, dlr=0.005, clip=50.0, print_every=10,
          teacher_forcing_ratio=lambda x: 1 if x < 10 else 0):
    # Configure training/optimization
    encoder_learning_rate = elr
    decoder_learning_ratio = dlr

    # Initialize models
    encoder = EncoderRNN(1, hidden_size, n_layers, dropout=dropout)
    decoder = LuongAttnDecoderRNN(
        attn_model, 1, hidden_size, n_layers, dropout=dropout)

    # Initialize optimizers and criterion
    encoder_optimizer = optimizer(
        encoder.parameters(), lr=encoder_learning_rate)
    decoder_optimizer = optimizer(
        decoder.parameters(), lr=decoder_learning_ratio)

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    # Begin!
    print_loss_total = 0
    step = 0
    while step < n_steps:
        step += 1
        # Get training data for this cycle
        batch_idx = np.random.randint(0, x.shape[1], batch_size)
        input_batches, target_batches = x[:,  batch_idx], y[:,  batch_idx]

        # Run the train function
        loss, _ = _train(
            input_batches, target_batches,
            encoder, decoder,
            encoder_optimizer, decoder_optimizer, criterion,
            teacher_forcing_ratio=teacher_forcing_ratio(step),
            clip=clip
        )
        # print(np.mean(np.square((output.data.cpu().numpy() - series[-20:,  batch_idx]))))
        # Keep track of loss
        print_loss_total += loss

        if step % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '(%d %d%%) %.4f' % (
                step, step / n_steps * 100, print_loss_avg)
            print(print_summary)
    return encoder, decoder


def evaluate(input_batch, lookahead, encoder, decoder):
    input_variable = Variable(torch.from_numpy(
        input_batch).unsqueeze(2).float())
    if USE_CUDA:
        input_variable = input_variable.cuda()

    # Set to not-training mode to disable dropout
    # encoder.train(False)
    # decoder.train(False)
    encoder.eval()
    decoder.eval()

    # Run sequences through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable)

    # Prepare input and output variables
    decoder_input = Variable(torch.zeros(
        1, input_batch.shape[1], 1))
    # decoder_input = input_variable[-1, :, :].unsqueeze(0)
    # Use last (forward) hidden state from encoder
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    all_decoder_outputs = Variable(
        torch.zeros(lookahead, input_variable.size()[1]))
    decoder_attentions = torch.zeros(lookahead, input_variable.size()[
                                     1], input_variable.size()[0])

    # Run through decoder one time step at a time
    for t in range(lookahead):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[t, :, :] = decoder_attn.squeeze(0).cpu().data
        all_decoder_outputs[t] = decoder_output[:, 0].cpu().data
        # Next input is current output
        decoder_input = decoder_output.unsqueeze(0)

    return all_decoder_outputs.data, decoder_attentions
