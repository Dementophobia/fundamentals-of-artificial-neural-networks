import numpy as np
from os import environ
from tensorflow import keras
from sys import argv

def input_impact(model):
    weights_pos   = model.layers[0].get_weights()[0][0]
    weights_speed = model.layers[0].get_weights()[0][1]

    print("\n------------------------------------------------------------------------------")
    print("\nWeights of neurons connected to the 'position' input:")
    print(weights_pos)
    print("\nWeights of neurons connected to the 'speed' input:")
    print(weights_speed)

    impact_pos   = sum([abs(weight) for weight in weights_pos])
    impact_speed = sum([abs(weight) for weight in weights_speed])

    sum_of_impact = impact_pos + impact_speed

    impact_pos   /= sum_of_impact
    impact_speed /= sum_of_impact

    print("\nNormalized sum of absolute weights (impact) of neurons connected to the 'position' input:")
    print(f"{impact_pos:.2f}")
    print("\nNormalized sum of absolute weights (impact) of neurons connected to the 'speed' input:")
    print(f"{impact_speed:.2f}")

def impact_on_action(model):
    weights_left    = np.array([model.layers[-1].get_weights()[0][neuron][0] for neuron in range(len(model.layers[-2].get_weights()[0]))])
    weights_neutral = np.array([model.layers[-1].get_weights()[0][neuron][1] for neuron in range(len(model.layers[-2].get_weights()[0]))])
    weights_right   = np.array([model.layers[-1].get_weights()[0][neuron][2] for neuron in range(len(model.layers[-2].get_weights()[0]))])

    print("\n------------------------------------------------------------------------------")
    print("\nWeights of neurons affecting the 'push left' action:")
    print(weights_left)
    print("\nWeights of neurons affecting the 'neutral' action:")
    print(weights_neutral)
    print("\nWeights of neurons affecting the 'push right' action:")
    print(weights_right)

    impact_left    = sum([abs(weight) for weight in weights_left])    / len(weights_left)
    impact_neutral = sum([abs(weight) for weight in weights_neutral]) / len(weights_neutral)
    impact_right   = sum([abs(weight) for weight in weights_right])   / len(weights_right)

    print("\nSum of absolute weights of neurons affecting the 'push left' action:")
    print(f"{impact_left:.2f}")
    print("\nSum of absolute weights of neurons affecting the 'neutral' action:")
    print(f"{impact_neutral:.2f}")
    print("\nSum of absolute weights of neurons affecting the 'push right' action:")
    print(f"{impact_right:.2f}")

def layer_impact(model, layer, verbose=False):
    size_of_layer = len(model.layers[layer].get_weights()[0])
    all_weights = [model.layers[layer].get_weights()[0][neuron] for neuron in range(size_of_layer)]
    sum_of_all_weights = sum([abs(weight) for neuron_weights in all_weights for weight in neuron_weights])
    
    if verbose:
        print("\n------------------------------------------------------------------------------")
        for neuron in range(len(all_weights)):
            print(f"\nWeights of neurons connected from neuron {neuron+1} on layer {layer} to neurons on layer {layer+1}:")
            print(all_weights[neuron])

            impact = sum([abs(weight) for weight in all_weights[neuron]]) / sum_of_all_weights

            print(f"\nNormalized sum of absolute weights of neurons connected from neuron {neuron+1} on layer {layer} to neurons on layer {layer+1}:")
            print(f"{impact:.2f}")
    
    print("\n------------------------------------------------------------------------------")
    print(f"\nImpact of neurons on layer {layer} connected to neurons on layer {layer+1}:")
    print("\nNeuron\tImpact")
    for neuron in range(len(all_weights)):
        impact = sum([abs(weight) for weight in all_weights[neuron]]) / sum_of_all_weights
        print(f"{neuron+1}\t{impact:.2f}")

def impacted_by(model, layer, input_id):
    size_of_input      = len(model.layers[0].get_weights()[0])
    size_of_prev_layer = len(model.layers[layer-1].get_weights()[0])
    
    if layer > 1:
        propageted_impact = np.zeros((2, size_of_prev_layer))
        propageted_impact[0] = impacted_by(model, layer-1, input_id)
        propageted_impact[1] = [1 - propageted_impact[0][pos] for pos in range(size_of_prev_layer)]
    else:
        propageted_impact = np.zeros((2, size_of_input))
        propageted_impact[0] = [1 if id == input_id else 0 for id in range(size_of_input)]
        propageted_impact[1] = [1 if id != input_id else 0 for id in range(size_of_input)]
        
    weights_prev_layer_id    = [abs(model.layers[layer-1].get_weights()[0][id] * propageted_impact[0][id]) for id in range(size_of_prev_layer)]
    weights_prev_layer_other = np.array([np.array(abs(model.layers[layer-1].get_weights()[0][id] * propageted_impact[1][id])) for id in range(size_of_prev_layer)])
    
    weights_prev_layer_id    = np.sum(weights_prev_layer_id, axis=0)
    weights_prev_layer_other = np.sum(weights_prev_layer_other, axis=0)
    
    sum_of_impacts = weights_prev_layer_id + weights_prev_layer_other
    weights_prev_layer_id /= sum_of_impacts

    print("\n------------------------------------------------------------------------------")
    print(f"\nEach neuron on layer {layer} gets impacted by the selected input parameter {input_id} 'Param' and all the other input parameters 'Other' propagated through the network as follows:")
    
    print("\nNeuron\tParam\tOther")
    
    for neuron in range(len(weights_prev_layer_id)):
        print(f"{neuron+1}\t{weights_prev_layer_id[neuron]:.2f}\t{1-weights_prev_layer_id[neuron]:.2f}")

    print(f"AVG:\t{sum(weights_prev_layer_id) / len(weights_prev_layer_id):.2f}\t{1 - sum(weights_prev_layer_id) / len(weights_prev_layer_id):.2f}")
    
    return weights_prev_layer_id

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=2)

if len(argv) <= 1:
    id = 0
else:
    id = argv[1]

model = keras.models.load_model(f"trained_model_{id}.h5")

input_impact(model)
impacted_by(model, 1, 0)
impacted_by(model, 3, 0)
layer_impact(model, 1)
layer_impact(model, 2)