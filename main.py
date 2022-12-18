import PySimpleGUI as sg

import tensorflow as tf

# define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, input_shape=(None,), activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# define the GUI layout
layout = [
    [sg.Text('Enter data (format: [x1, x2, ...], y)')],
    [sg.Input(key='data')],
    [sg.Button('Add'), sg.Button('Train')],
    [sg.Text('Enter data to predict (format: [x1, x2, ...])')],
    [sg.Input(key='predict')],
    [sg.Button('Predict')],
    [sg.Output(key='output')]
]

# create the GUI window
window = sg.Window('Neural Network', layout)

# define the training data
X = []
y = []

# main loop to process GUI events
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

    # process "Add" button event
    if event == 'Add':
        data = eval(values['data'])
        X.append(data[0])
        y.append(data[1])

    # process "Train" button event
    if event == 'Train':
        model.fit(X, y, epochs=5)
        window['output'].update('Model trained!')

    # process "Predict" button event
    if event == 'Predict':
        prediction = model.predict(eval(values['predict']))
        window['output'].update(prediction)

window.close()
