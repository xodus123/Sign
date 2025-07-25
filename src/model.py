from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2

# Dense 모델 정의
def build_experiment_model_1(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(2048, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_experiment_model_2(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(2048, input_dim=input_dim, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_experiment_model_3(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_experiment_model_4(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_experiment_model_5(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_experiment_model_6(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_experiment_model_7(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(2048, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# CNN 모델 정의
def build_experiment_model_8(input_shape, output_dim, dropout_rate=0.3):
    model = Sequential()
    model.add(Conv2D(32, (5, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_experiment_model_9(input_shape, output_dim, dropout_rate=0.4):
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))     

    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(dropout_rate))   # Dense 앞에 강하게 Dropout

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 모델 선택 딕셔너리 (함수 호출 파라미터 통일 래퍼 적용)
model_dict = {
    'exp1': lambda input_dim, output_dim, **kwargs: build_experiment_model_1(input_dim, output_dim),
    'exp2': lambda input_dim, output_dim, **kwargs: build_experiment_model_2(input_dim, output_dim),
    'exp3': lambda input_dim, output_dim, **kwargs: build_experiment_model_3(input_dim, output_dim),
    'exp4': lambda input_dim, output_dim, **kwargs: build_experiment_model_4(input_dim, output_dim),
    'exp5': lambda input_dim, output_dim, **kwargs: build_experiment_model_5(input_dim, output_dim),
    'exp6': lambda input_dim, output_dim, **kwargs: build_experiment_model_6(input_dim, output_dim),
    'exp7': lambda input_dim, output_dim, **kwargs: build_experiment_model_7(input_dim, output_dim),
    'exp8': lambda input_dim, output_dim, input_shape: build_experiment_model_8(input_shape, output_dim),
    'exp9': lambda input_dim, output_dim, input_shape: build_experiment_model_9(input_shape, output_dim)
}

# 모델 래퍼 (Dense/CNN 자동구분 + 실험자동화 최적)
def get_model(model_name, X, y, **kwargs):
    if model_name in ['exp8', 'exp9']:
        if 'input_shape' not in kwargs:
            raise ValueError('CNN 모델은 input_shape=(h, w, c) 지정 필요!')
        return model_dict[model_name](None, output_dim=y.shape[1], **kwargs)
    else:
        return model_dict[model_name](input_dim=X.shape[1], output_dim=y.shape[1], **kwargs)
