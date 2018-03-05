import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.python import debug as tf_debug

START_ID=0
PAD_ID=1
END_ID=2

# Globalgraph = tf.Graph()
# with Globalgraph.as_default():
# 	sess = tf.Session()


tf.app.flags.DEFINE_integer("batchSize", 100,"Batch size.")
tf.app.flags.DEFINE_integer("input_max_length", 5, "Maximum input sequence length.")
tf.app.flags.DEFINE_integer("output_max_length", 7, "Maximum output sequence length.")
tf.app.flags.DEFINE_integer("rnn_size", 100, "RNN unit size.")
tf.app.flags.DEFINE_integer("attention_size", 100, "Attention size.")
tf.app.flags.DEFINE_integer("beamWidth", 2, "Width of beam search .")
tf.app.flags.DEFINE_float("learningRate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Maximum gradient norm.")
tf.app.flags.DEFINE_string("log_dir", "./log", "Log directory")
tf.app.flags.DEFINE_string("data_path", "./data/convex_hull_5_test.txt", "Data path.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100, "frequence to do per checkpoint.")
FLAGS = tf.app.flags.FLAGS

class PointerWrapper(tf.contrib.seq2seq.AttentionWrapper):
  def __init__(self,cell,attention_size,memory,initial_cell_state=None,name=None):
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(attention_size, memory, probability_fn=lambda x: x )
    cell_input_fn=lambda input, attention: input
    super(PointerWrapper, self).__init__(cell,
                                         attention_mechanism=attention_mechanism,
                                         attention_layer_size=None,
                                         alignment_history=False,
                                         cell_input_fn=cell_input_fn,
                                         output_attention=True,
                                         initial_cell_state=initial_cell_state,
                                         name=name)
  @property
  def output_size(self):
    return self.state_size.alignments
  def call(self, inputs, state):
    _, next_state = super(PointerWrapper, self).call(inputs, state)
    return next_state.alignments, next_state


class Pointer(object):
  def __init__(self, batchSize=100, input_max_length=5, output_max_length=7, rnn_size=100, attention_size=100, beamWidth=2, learningRate=0.001, max_gradient_norm=5):
    self.batchSize = batchSize
    self.input_max_length = input_max_length
    self.output_max_length = output_max_length
    self.learningRate = learningRate

    #To add 3 special values of start, pad and end for sequence
    self.vocabulary = input_max_length+3      
    self.global_step = tf.Variable(0,trainable=False)
    cell = tf.contrib.rnn.LSTMCell

    # last 2 factor is because I assume that this network is for coordinate points i.e some input of form (x,y)
    self.inputs = tf.placeholder(tf.float32, shape=[self.batchSize,self.input_max_length,2], name="inputs")
    self.outputs = tf.placeholder(tf.int32, shape=[self.batchSize,self.output_max_length+1], name="outputs")
    self.encoder_input_wt = tf.placeholder(tf.int32, shape=[self.batchSize,self.input_max_length], name="encoder_input_wt")
    self.decoder_input_wt = tf.placeholder(tf.int32, shape=[self.batchSize,self.output_max_length], name="decoder_input_wt")

    encoder_in_length=tf.reduce_sum(self.encoder_input_wt,axis=1)
    decoder_in_length=tf.reduce_sum(self.decoder_input_wt,axis=1)

    embedding_tok = tf.get_variable("embedding_tok", [3,2], tf.float32,tf.contrib.layers.xavier_initializer())

    # shape=[1,3,2]
    tempExp = tf.expand_dims(embedding_tok,0)
    # shape=[batchSize,3,2]
    tempexp2 = tf.tile(tempExp,[self.batchSize,1,1])
    # shape=[batchSize,vocabularySize,2]
    total_table = tf.concat([tempexp2,self.inputs],axis=1)
    # shape= batchSize*[vocabularySize,2]
    total_table_list = tf.unstack(total_table, axis=0)
    # shape= (output_max_length+1)*[batchSize]
    outputList = tf.unstack(self.outputs, axis=1)
    # shape= [batchSize,output_max_length]
    self.target = tf.stack(outputList[1:], axis=1)

    ids_dec = tf.unstack(tf.expand_dims(tf.stack(outputList[:-1],axis=1),2),axis=0)
    ids_enc = [tf.expand_dims(tf.range(2,self.vocabulary),1)]*self.batchSize

    encoder_inputs = []
    decoder_inputs = []
    for i in range(self.batchSize):
      encoder_inputs.append(tf.gather_nd(total_table_list[i], ids_enc[i]))
      decoder_inputs.append(tf.gather_nd(total_table_list[i], ids_dec[i]))

    encoder_inputs = tf.stack(encoder_inputs, axis=0) 
    decoder_inputs = tf.stack(decoder_inputs, axis=0) 

    encoder_fwd_cell = cell(rnn_size)
    encoder_bwd_cell = cell(rnn_size)

    memory,_ = tf.nn.bidirectional_dynamic_rnn(encoder_fwd_cell,encoder_bwd_cell, encoder_inputs, encoder_in_length, dtype=tf.float32)
    memory = tf.concat(memory,2)

    ptr_cell = PointerWrapper(cell(rnn_size),attention_size,memory)
    dec_cell = ptr_cell

    # max length of sequence in this batch
    curBatchMaxLen = tf.reduce_max(decoder_in_length)

    helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, decoder_in_length)
    # basic decoder 
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, dec_cell.zero_state(self.batchSize,tf.float32))
    # decode
    outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True)
    # logits
    logits = outputs.rnn_output
    # predictions -> ids
    self.predicted_ids_with_logits = tf.nn.top_k(logits)
    # To make logits same as target to compare easily

    logits = tf.concat([logits, tf.ones([self.batchSize,self.output_max_length-curBatchMaxLen,self.input_max_length+1])], axis=1)
    # print(curBatchMaxLen)

    self.real_target = (self.target-2)*self.decoder_input_wt
    # losses
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.real_target, logits = logits)
    # net loss
    self.loss = tf.reduce_sum(losses*tf.cast(self.decoder_input_wt,tf.float32))/self.batchSize
    # print(self.loss)
    params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, params)
    # clip gradients
    clipped_grads,_ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    # optimization
    optimizer = tf.train.AdamOptimizer(self.learningRate)
    self.update = optimizer.apply_gradients(zip(clipped_grads,params), global_step=self.global_step)
    tf.summary.scalar('loss',self.loss)
    for p in params:
      tf.summary.histogram(p.op.name,p)
    for p in gradients: 
      tf.summary.histogram(p.op.name,p)
    self.summary_op = tf.summary.merge_all()
    self.saver = tf.train.Saver(tf.global_variables())

  def step(self, session, inputs, encoder_input_wt, outputs=None, decoder_input_wt=None):
    input_feed = {}
    input_feed[self.inputs] = inputs
    input_feed[self.encoder_input_wt] = encoder_input_wt
    input_feed[self.outputs] = outputs
    input_feed[self.decoder_input_wt] = decoder_input_wt

    output_feed = [self.update, self.summary_op, self.loss, self.predicted_ids_with_logits, self.real_target]
    outputs = session.run(output_feed, input_feed)
    return outputs[1],outputs[2],outputs[3],outputs[4]

    

class MainModel(object):
  def __init__(self):
    self.graph = tf.Graph()
    # self.graph = Globalgraph
    with self.graph.as_default():
      # self.sess = tf_debug.LocalCLIDebugWrapperSession(sess)
      self.sess = tf.Session()
    # print("HI")
    self.build_model()
    self.read_data()

  def read_data(self):
    with open(FLAGS.data_path,'r') as file:
      lines = file.readlines()
      inputs = []
      encoder_input_wt = []
      outputs = []
      decoder_input_wt = []
      
      for line in lines:
        inp, outp = line[:-2].split(' output ')
        inp = inp.split(' ')
        outp = outp.split(' ')
        encoderInput = []
        for t in inp:
          encoderInput.append(float(t))
        # Input has both x and y coordinates so we take (x,y) as 1 so divide by 2
        temp_length = len(encoderInput)//2   
        encoderInput += [0]*((FLAGS.input_max_length - temp_length)*2) 
        encoderInput = np.array(encoderInput).reshape([-1,2])
        inputs.append(encoderInput)
        weight = np.zeros(FLAGS.input_max_length)
        weight[:temp_length]=1
        encoder_input_wt.append(weight)
   
        output=[START_ID]
        for i in outp:
          output.append(int(i)+2)
        output.append(END_ID)
        temp_len_decode = len(output)-1
        output += [PAD_ID]*(FLAGS.output_max_length-temp_len_decode)
        output = np.array(output)
        outputs.append(output)
        weight = np.zeros(FLAGS.output_max_length)
        weight[:temp_len_decode]=1
        decoder_input_wt.append(weight)
      
      self.inputs = np.stack(inputs)
      self.encoder_input_wt = np.stack(encoder_input_wt)
      self.outputs = np.stack(outputs)
      self.decoder_input_wt = np.stack(decoder_input_wt)
      print("Load inputs:            " +str(self.inputs.shape))
      print("Load encoder_input_wt: " +str(self.encoder_input_wt.shape))
      print("Load outputs:           " +str(self.outputs.shape))
      print("Load decoder_input_wt: " +str(self.decoder_input_wt.shape))

  def get_batch(self):
    data_size = self.inputs.shape[0]
    sample = np.random.choice(data_size,FLAGS.batchSize,replace=True)
    return self.inputs[sample],self.encoder_input_wt[sample], self.outputs[sample], self.decoder_input_wt[sample]

  def build_model(self):
    with self.graph.as_default():
      # Build model
      self.model = Pointer(batchSize=FLAGS.batchSize,
                    input_max_length=FLAGS.input_max_length,
                    output_max_length=FLAGS.output_max_length,
                    rnn_size=FLAGS.rnn_size,
                    attention_size=FLAGS.attention_size,
                    beamWidth=FLAGS.beamWidth,
                    learningRate=FLAGS.learningRate,
                    max_gradient_norm=FLAGS.max_gradient_norm)

      self.writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',self.sess.graph)
      ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
      if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Load model parameters from %s" % ckpt.model_checkpoint_path)
        self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
      else:
        print("Created model with fresh parameters.")
        self.sess.run(tf.global_variables_initializer())

  def train(self):
    step_time = 0.0
    loss = 0.0
    current_step = 0

    while True:
      start_time = time.time()
      inputs,encoder_input_wt, outputs, decoder_input_wt = self.get_batch()
      summary, step_loss, predicted_ids_with_logits, target = self.model.step(self.sess, inputs, encoder_input_wt, outputs, decoder_input_wt)
      # print(logits)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      if current_step % FLAGS.steps_per_checkpoint == 0:
        with self.sess.as_default():
          gstep = self.model.global_step.eval()
        print ("global step %d step-time %.2f loss %.2f" % (gstep, step_time, loss))

        self.writer.add_summary(summary, gstep)

        sample = np.random.choice(FLAGS.batchSize,1)[0]
        print("-"*50)
        print("Predict: "+str(np.array(predicted_ids_with_logits[1][sample]).reshape(-1)))
        print("Target : "+str(target[sample]))
        print("-"*50)  
        checkpoint_path = os.path.join(FLAGS.log_dir, "convex_hull.ckpt")
        self.model.saver.save(self.sess, checkpoint_path, global_step=self.model.global_step)
        step_time, loss = 0.0, 0.0

  def eval(self):
    inputs,encoder_input_wt, outputs, decoder_input_wt = self.get_batch()
    predicted_ids = self.model.step(self.sess, inputs, encoder_input_wt)    
    print("="*20)
    for i in range(FLAGS.batchSize):
      print("* %dth sample target: %s" % (i,str(outputs[i,1:]-2)))
      for predict in predicted_ids[i]:
        print("prediction: "+str(predict))       
    print("="*20)

  def run(self):
    self.train()



def main(_):
  Main_Model = MainModel()
  Main_Model.run()

if __name__ == "__main__":
  tf.app.run()
