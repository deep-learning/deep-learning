from __future__ import print_function

import mxnet as mx
from mxnet import nd, gluon, autograd

data_ctx = mx.cpu()
model_ctx = mx.cpu()

num_inputs = 2
num_outputs = 1
num_examples = 10000


def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2


X = nd.random_normal(shape=(num_examples, num_inputs))
noise = 0.01 * nd.random_normal(shape=(num_examples,))
y = real_fn(X) + noise

batch_size = 4
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                   batch_size=batch_size, shuffle=True)

# net = gluon.nn.Dense(units=1, in_units=2)
# net = gluon.nn.Sequential()
# net.add(gluon.nn.Dense(1))
net = gluon.nn.Dense(1)

print(net.weight)
print(net.bias)
print(net.collect_params())
print(type(net.collect_params()))

# Deferred Initialization
# the actual initialization is deferred until we make a first forward pass
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)
# net.initialize()

example_data = nd.array([[4, 7]])
print(net(example_data))

print(net.weight.data())
print(net.bias.data())

square_loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.0001})
epochs = 10
loss_sequence = []
num_batches = num_examples / batch_size

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        # Calculate gradients
        loss.backward()

        # update model parameters
        trainer.step(batch_size)

        cumulative_loss += nd.mean(loss).asscalar()
    print("Epoch %s, loss: %s" % (e, cumulative_loss / num_examples))
    loss_sequence.append(cumulative_loss)

import matplotlib.pyplot as plt

plt.figure(num=None,figsize=(8, 6))
plt.plot(loss_sequence)

# Adding some bells and whistles to the plot
plt.grid(True, which="both")
plt.xlabel('epoch',fontsize=14)
plt.ylabel('average loss',fontsize=14)
plt.show()


# dense = net[0]
# dense.weight.data()
# dense.bias.data()

# ParameterDict
params = net.collect_params()
for param in params.values():
    print(param.name, param.data())

