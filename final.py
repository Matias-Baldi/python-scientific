import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# %% Datos de pestaneo
signals = pd.read_csv('data/pestaneos.dat', delimiter=' ', names=['timestamp', 'counter', 'eeg', 'attention', 'meditation', 'blinking'])
data = signals.values

# Verificar nulos y duplicados
print(signals.isnull().sum())  # No hay nulos
signals = signals.drop_duplicates()
print(signals.shape)

# %% Datos de baseline
signalsb = pd.read_csv('data/baseline.dat', delimiter=' ', names=['timestamp', 'counter', 'eeg', 'attention', 'meditation', 'blinking'])
datab = signalsb.values

# Verificar nulos y duplicados
print(signalsb.isnull().sum())  # No hay nulos
signalsb = signalsb.drop_duplicates()
print(signalsb.shape)

# %% Graficar señales originales

# Pestaneos
eeg = data[:, 2]
plt.plot(eeg, 'r', label='EEG')
plt.xlabel('t')
plt.ylabel('eeg(t)')
plt.title('EEG Signal (Pestañeos)')
plt.ylim([-2000, 2000])
plt.xlim([0, len(eeg)])
plt.show()

sns.set(style="darkgrid")
sns.lineplot(x="timestamp", y="eeg", hue="attention", data=signals)
plt.show()

# Baseline
eegb = datab[:, 2]
plt.plot(eegb, 'r', label='EEG')
plt.xlabel('t')
plt.ylabel('eeg(t)')
plt.title('EEG Signal (Baseline)')
plt.ylim([-2000, 2000])
plt.xlim([0, len(eegb)])
plt.show()

sns.lineplot(x="timestamp", y="eeg", hue="attention", data=signalsb)
plt.show()

# %% Filtro temporal - Moving average
windowlength = 10
avgeeg = np.convolve(eeg, np.ones((windowlength,)) / windowlength, mode='same')
avgeegb = np.convolve(eegb, np.ones((windowlength,)) / windowlength, mode='same')

# Graficar señales suavizadas
plt.plot(avgeeg, 'r', label='EEG')
plt.xlabel('t')
plt.ylabel('eeg(t)')
plt.title('Smoothed EEG Signal (Pestañeos)')
plt.ylim([-2000, 2000])
plt.xlim([0, len(avgeeg)])
plt.show()

plt.plot(avgeegb, 'r', label='EEG')
plt.xlabel('t')
plt.ylabel('eeg(t)')
plt.title('Smoothed EEG Signal (Baseline)')
plt.ylim([-2000, 2000])
plt.xlim([0, len(avgeegb)])
plt.show()

# %% Z-score normalization
def z_score_norm(arr):
    mean_ = np.mean(arr)
    std_ = np.std(arr)
    return [(i - mean_) / std_ for i in arr]

eeg_zscore = z_score_norm(eeg)
plt.plot(eeg_zscore, 'r', label='EEG')
plt.xlabel('t')
plt.ylabel('eeg(t)')
plt.title('z-score EEG Signal (Pestañeos)')
plt.ylim([-2000, 2000])
plt.xlim([0, len(eeg_zscore)])
plt.show()

# %% Filtro de paso de banda
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Parámetros del filtro
fs = 128.0
lowcut = 0.5
highcut = 40.0

baseline_filtered = butter_bandpass_filter(avgeegb, lowcut, highcut, fs)
blink_filtered = butter_bandpass_filter(avgeeg, lowcut, highcut, fs)

# Extraer características
def extract_features(data, threshold):
    peaks, _ = find_peaks(data, height=threshold)
    num_peaks = len(peaks)
    return num_peaks

threshold = 0.5  # Ajustar según sea necesario
baseline_features = extract_features(baseline_filtered, threshold)
blink_features = extract_features(blink_filtered, threshold)

# Crear etiquetas: 0 para baseline, 1 para pestañeos
X = np.array([[baseline_features], [blink_features]])
y = np.array([0, 1])

# Crear el clasificador SVM
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# Hacer predicciones
y_pred = clf.predict(X)

# Evaluar el clasificador
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))
print("Classification Report:")
print(classification_report(y, y_pred))

# Graficar señales filtradas
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(baseline_filtered)
plt.title('Baseline Filtrada')
plt.subplot(2, 1, 2)
plt.plot(blink_filtered)
plt.title('Pestañeo Filtrado')
plt.tight_layout()
plt.show()