
logit = model.predict(np.arange(adj.shape[0]))
predict = softmax(logit)
e0, e1 = sp.tril(attacker.A).nonzero()
prob_e0, prob_e1 = tf.gather(predict, e0), tf.gather(predict, e1)
kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
divergence_after = kl(prob_e1, (prob_e0+prob_e1)/2.) + kl(prob_e0, (prob_e0+prob_e1)/2.)
plt.hist(divergence_after.numpy())
