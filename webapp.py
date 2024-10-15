import streamlit as st
import pandas as pd


# misc stuff
data_csv = 'gym_members_exercise_tracking.csv'
example_data = pd.read_csv(data_csv).head(10)

with open(data_csv, 'r') as file:
    downloadable_file = file.read()
    



# title of web app
st.markdown("<h1 style='text-align: center;'>Calorie Burning Predictions</h1>", unsafe_allow_html=True)

# data section
st.subheader('The Data')

st.write('The dataset was found on kaggle. Here are the first few lines from the dataframe.')


st.download_button(
    label='Click here to download the data',
    data=downloadable_file,
    file_name='gym_membership_data.csv',
    mime='text/csv'
)


st.dataframe(example_data, hide_index=True, height=400)

# data analysis
st.subheader('Analysis')
analysis = 'My goal was for this project to predict the calories burned. Exploring the dataset, I identified features which seemed to influence the predictions.\
            The features I selected were age, gender, weight, height, heart rate, session duration, workout type, fat percentage and workout duration. I eliminated \
            unimportant features such as water consumption and workout frequency.'
st.write(analysis)

# data preparation
st.subheader('Data Preprocessing')

data_preprocessing_text_1 = 'The first step was to import the necessary libraries for this project.'
st.write(data_preprocessing_text_1)

preprocesing_code_1 = '''
    import pandas as pd
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    '''
st.code(preprocesing_code_1, language='python')

data_preprocessing_text_2 = 'The second step was to create a features and target dataframe. The unimportant features and target were dropped from the original data.'
st.write(data_preprocessing_text_2)

preprocesing_code_2 = '''
    csv_file = 'gym_members_exercise_tracking.csv'
    data = pd.read_csv(csv_file)

    features = data.drop(['Calories_Burned','Water_Intake (liters)', 'Workout_Frequency (days/week)'], axis=1)
    target = data['Calories_Burned']
    '''
st.code(preprocesing_code_2, language='python')

data_preprocessing_text_3 = 'Next, non numerical features must be encoded so they can be fed to the model. Training and testing datasets must be created and scaled.'
st.write(data_preprocessing_text_3)

preprocesing_code_3 = '''
    encoded_features = pd.get_dummies(features, columns=['Workout_Type', 'Gender'])

    X_train, X_test, y_train, y_test = train_test_split(encoded_features, target, test_size=.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    '''
st.code(preprocesing_code_3, language='python')

# model architecture
st.subheader('Model Architecture')

architecture_text_1 = 'I chose to create a neural network. After experimenting I settled on 12 layers and a standard batch size of 32. \
            After each dense layer I added batch normalization with the activation function relu and a dropout of 0.25 to help prevent overfitting.'
st.write(architecture_text_1)

model_architecture_code_1 = '''
    model = tf.keras.Sequential([
        tf.keras.layers.Input(X_train_scaled.shape[1]),
    
        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(.25),
        
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(.25),
        
        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(1)
    ])
    '''
st.code(model_architecture_code_1, language='python')

architecture_text_2 = 'To further prevent overfitting, I implemented early stopping. For my optimizer, I used the Adam function with a learning rate of 0.002 \
            which seemed to perform better than the standard learning rate of 0.001. The loss I decided upon was mean squared error and the metric I used \
            was mean absolute error.'
st.write(architecture_text_2)

model_architecture_code_2 = '''
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=5,
                                                      restore_best_weights=True
                                                      )

    optimizer = tf.keras.optimizers.Adam(learning_rate=.002)

    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae']
                  )
    '''
st.code(model_architecture_code_2, language='python')

architecture_text_3 = 'The last step is to fit the model to the data. Since I did not directly create a validation set, I used the argument validation_split \
            to use 20\% of the training data as validation data. Since I implemented early stopping I decided to run the model for 500 epochs to experiment.'
st.write(architecture_text_3)       

model_architecture_code_3 = '''
model.fit(X_train_scaled,
          y_train,
          validation_split=.2,
          validation_steps=10,
          shuffle=True,
          epochs=500,
          callbacks=[early_stopping],
          verbose=2
          )
    '''     
st.code(model_architecture_code_3, language='python')

# results
st.subheader('Results')

results_text_1 = 'After optimizing the model the best I could, my final results were a test loss of ~1217.6 calories and a test mean \
            absolute error of ~22.3 calories.'
st.write(results_text_1)

model_results_code_1 = '''
    results = model.evaluate(X_test_scaled, y_test)
    print("Test loss:", results[0])
    # 1217.6
    print("Test mae:", results[1])
    # 22.3
    '''
st.code(model_results_code_1, language='python')

# future improvements
st.subheader('Future Improvements')

future_improvements = 'In the future, I would like to increase the accuracy further by doing a bit more data preprocessing and by improving my model \
            architecture. I think with enough experimentation I could get the mean absolute error below 10 calories.'
st.write(future_improvements)

with st.sidebar:
    st.header('Page Navigation')
    st.markdown('[The Data](#the-data)')
    st.markdown('[Analysis](#analysis)')
    st.markdown('[Data Preprocessing](#data-preprocessing)')
    st.markdown('[Model Architecture](#model-architecture)')
    st.markdown('[Results](#results)')
    st.markdown('[Future Improvements](#future-improvements)')
    
    