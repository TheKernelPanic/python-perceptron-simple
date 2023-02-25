import numpy
import matplotlib.pyplot as plt

# Programación de una neurona perceptrón a través de datos linealmente separables

# | Superado examen 1 | Superado examen 2 | Admitido |
# |         1         |         0         |    0     |
# |         1         |         1         |    1     |
# |         0         |         1         |    0     |
# |         0         |         0         |    0     |

# Observaciones (Examenes 1 y 2)
input_observations = numpy.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Datos de aprendizaje
predictions = numpy.array([[0], [0], [0], [1]])

# Sesgo
bias = 1

# Definir pesos
numpy.random.seed(1)
min_limit = -1
max_limit = 1

w11 = (max_limit - min_limit) * numpy.random.random() + min_limit
w21 = (max_limit - min_limit) * numpy.random.random() + min_limit
wb = 0

# Hiperparámetros para el aprendizaje
learn_rate = 0.1
epochs = 10000

# Lista de valores para la representación gráfica
graphic_mse = []


# Cálculo de la suma ponderada
def weighted_sum(X1, W11, X2, W21, B, WB):
    return B * WB + (X1 * W11 + X2 * W21)


# Función de activación de tuipo sigmoide
def activation_sigmoid(__weighted_sum_value):
    return 1 / (1 + numpy.exp(-__weighted_sum_value))


# Función de activación Relu
def activation_relu(__weighted_sum_value):
    return max(0, __weighted_sum_value)


# Cálculo del error lineal
def lineal_error(expect_value, __prediction_value):
    return expect_value - __prediction_value


# Cálculo del gradiente
def calculate_gradient(input_value, prediction, error):
    return -1 * error * prediction * (1 - prediction) * input_value


# Cálculo del valor de ajustes del peso
def calculate_weight_setting_value(gradient_value, __learn_rate):
    return gradient_value * __learn_rate


# Cálculo del valor del peso
def calculate_new_weight(weight_value, setting_value):
    return weight_value - setting_value


# Error medio cuadrático (MSE)
def calculate_mse(__predictions_made, __predictions_expected):
    i = 0
    __sum = 0
    for _ in __predictions_expected:
        diff = __predictions_expected[i][0] - __predictions_made[i][0]
        __sum = __sum + (diff * diff)
        i = i + 1
    return 1 / (len(__predictions_expected)) * __sum


# Calculo de precisión
def calculate_accuracy(__predictions_made, __predictions_expected):
    __predictions_expected = numpy.array(__predictions_expected)
    __predictions_made = numpy.array(__predictions_made)
    return (__predictions_made.round() == __predictions_expected).astype(int).sum() / __predictions_expected.shape[0]


# Fase de apredizaje
for epoch in range(0, epochs):
    print("EPOCH (" + str(epoch) + "/" + str(epoch) + ")")
    predictions_made_during_epoch = []
    observations_index = 0
    for observation in input_observations:
        # Carga de la capa de entrada
        x1 = observation[0]
        x2 = observation[1]

        # Valor de predicción esperado
        expected_value = predictions[observations_index][0]

        # (1) Cálculo de la suma ponderada
        weighted_sum_value = weighted_sum(x1, w11, x2, w21, bias, wb)

        # (2) Aplicar función de activación
        predicted_value = activation_sigmoid(weighted_sum_value)

        # (3) Cálculo del error
        error_value = lineal_error(expected_value, predicted_value)

        # Actualización del peso 1
        # Cálculo del gradiente del valor de ajuste y del peso nuevo
        gradient_W11 = calculate_gradient(x1, predicted_value, error_value)
        value_setting_W11 = calculate_weight_setting_value(gradient_W11, learn_rate)
        w11 = calculate_new_weight(w11, value_setting_W11)

        # Actualización del peso 2
        gradient_W21 = calculate_gradient(x2, predicted_value, error_value)
        value_setting_W21 = calculate_weight_setting_value(gradient_W21, learn_rate)
        w21 = calculate_new_weight(w21, value_setting_W21)

        # Actualización del peso del sesgo
        gradient_Wb = calculate_gradient(bias, predicted_value, error_value)
        value_setting_Wb = calculate_weight_setting_value(gradient_Wb, learn_rate)
        wb = calculate_new_weight(wb, value_setting_Wb)

        print(
            "EPOCH (" + str(epoch) + "/" + str(epochs) + ") - Observation: " + str(observations_index + 1) + "/" + str(
                len(input_observations)))

        # Almacenamiento de la predicción realizada
        predictions_made_during_epoch.append([predicted_value])

        observations_index = observations_index + 1

    MSE = calculate_mse(predictions_made_during_epoch, predictions)
    accuracy = calculate_accuracy(predictions_made_during_epoch, predictions)

    graphic_mse.append(MSE)

    print("MSE: [" + str(MSE) + "]")
    print("Accuracy: [" + str(accuracy) + "]")

# Mostrar curva del MSE
plt.plot(graphic_mse)
plt.ylabel('MSE')
plt.show()

print("Cantidad de épocas: " + str(epochs))
print("Pesos finales")
print("W11: " + str(w11))
print("W21: " + str(w21))
print("Wb: " + str(wb))

# Realizar predicción con los valores obtenidos
x1 = 0
x2 = 0

print("Predicción")
weighted_sum_value = weighted_sum(x1, w11, x2, w21, bias, wb)
predicted_value = activation_sigmoid(weighted_sum_value)

print("Predicción del [" + str(x1) + "," + str(x2) + "]")
print("Predicción = " + str(predicted_value))
