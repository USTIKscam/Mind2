import tensorflow as tf
import model

def top_k_logits(logits, k):
    if k == 0:
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )

def top_p_logits(logits, p):
    batch, _ = logits.shape.as_list()
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack([
        tf.range(0, batch),
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values[:, tf.newaxis], # ensure min_values is broadcastable
        tf.ones_like(logits) * -1e10,
        logits,
    )

def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=1):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams_arg, tokens, past_arg=None):
        lm_output = model.model(hparams=hparams_arg, X=tokens, past=past_arg, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams_arg.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams_arg, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        def body(past_arg, prev_arg, output_arg):
            next_outputs = step(hparams, prev_arg, past=past_arg)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, p=top_p)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            
            current_presents = next_outputs['presents']
            if past_arg is None:
                past_new = current_presents
            else:
                past_new = tf.concat([past_arg, current_presents], axis=-2) # check axis, original might be wrong if past_shape has sequence in 4th pos
                                                                            # past_shape = [batch, n_layer, 2, n_head, sequence, features]
                                                                            # presents_shape = [batch, n_layer, 2, n_head, 1, features] (for single token step)
                                                                            # original axis=-2 is correct for tf.stack([k,v]) giving shape [B,L,2,H,S,F], dim S is -2.


            return [
                past_new,
                samples,
                tf.concat([output_arg, samples], axis=1)
            ]

        past_init, prev_init, output_init = body(None, context, context)

        def cond(*args):
            return True

        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=[
                past_init,
                prev_init,
                output_init
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size, sequence=None)), # Allow sequence dim to grow
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        return tokens